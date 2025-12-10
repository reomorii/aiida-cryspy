from aiida.orm import Int,Dict,List,Code,ArrayData,RemoteData,FolderData,load_group,Group
from aiida.engine import WorkChain,calcfunction,ToContext,while_,append_
from aiida.plugins import DataFactory
from ase.units import GPa  # 圧力の単位（GPa）をASEの内部単位(eV/Å^3)に変換
import os
import uuid

from cryspy.job import ctrl_job
from aiida_mlip.data.model import ModelData

StructureCollectionData = DataFactory("aiida_cryspy.structurecollection")
PandasFrameData = DataFactory("aiida_cryspy.dataframe")
RinData = DataFactory("aiida_cryspy.rin_data")
EAData = DataFactory("aiida_cryspy.ea_data")
StructureData = DataFactory("core.structure")


class optimization_WorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code, help="label of your code")
        spec.input("structure", valid_type=StructureData, help="selected structure for optimization")
        spec.input("parameters", valid_type=Dict)
        spec.input("options", valid_type=Dict, default=Dict, help="metadata.options")

        spec.output("remote_folder", valid_type=RemoteData, help="remote folder of the workchain")
        spec.output("retrieved", valid_type=FolderData, help="retrieved data from the workchain")
        spec.output("structure", valid_type=StructureData, help="optimized structure from the workchain")
        spec.output("array", valid_type=ArrayData, help="array data from the workchain")
        spec.output("parameters", valid_type=Dict, help="output parameters from the workchain")

        spec.outline(
            cls.submit_workchains,
            cls.inspect_workchains
        )



    def submit_workchains(self):
        """
        AiiDAを使って、MattersimによるBaTiO3のセル最適化を実行するための
        最終修正版スクリプト。
        """

        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = self.inputs.structure
        builder.parameters = self.inputs.parameters
        builder.metadata.options = self.inputs.options.get_dict()
        # builder.metadata.options.max_wallclock_seconds = 1 * 30 * 60
        builder.metadata.options.parser_name = "ase.ase"
        builder.metadata.options.additional_retrieve_list = ["opt.traj", "opt_struc.vasp"]
        # submit workchain
        future = self.submit(builder)
        return ToContext(my_future=future)


    def inspect_workchains(self):
        #sleepを入れて並列を確認
        calculations = self.ctx.my_future

        if "remote_folder" in calculations.outputs:
            self.out("remote_folder", calculations.outputs.remote_folder)
        if "array" in calculations.outputs:
            self.out("array", calculations.outputs.array)
        if "retrieved" in calculations.outputs:
            self.out("retrieved", calculations.outputs.retrieved)
        if "parameters" in calculations.outputs:
            self.out("parameters", calculations.outputs.parameters)
        if "structure" in calculations.outputs:
            self.out("structure", calculations.outputs.structure)


@calcfunction
def pack_results(**kwargs):
    """
    複数の計算結果ノードを受け取り、エネルギーと構造の辞書（pymatgen.as_dict()）
    をまとめた一つのDictノードを返す。
    """
    final_results = {}

    # 'id_0'のようなキーでデータをグループ化するための辞書
    grouped_data = {}
    for key, node in kwargs.items():
        # キーの名前からID部分 ("id_0"など) を取り出す
        id_ = '_'.join(key.split('_')[1:])

        # 辞書がなければ作成
        if id_ not in grouped_data:
            grouped_data[id_] = {}

        # キーのプレフィックスでresultかstructureかを判断
        if key.startswith("parameters_"):
            grouped_data[id_]["parameters"] = node
        elif key.startswith("structure_"):
            grouped_data[id_]["structure"] = node

    # グループ化されたデータを処理
    for id_, data in grouped_data.items():
        # parametersノードからエネルギーを取得
        energy = data["parameters"].get_dict().get("total_energy")

        # structureノードからpymatgenオブジェクトを取得し、JSON互換の辞書に変換
        struc_dict = data["structure"].get_pymatgen().as_dict()

        final_results[id_] = {"energy": energy, "structure": struc_dict}

    # Python辞書をAiiDAのDictノードとして返す
    return Dict(dict=final_results)


class multi_structure_optimize_WorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("initial_structures_group_pk", valid_type=Int, help="PK of the group containing initial structures for optimization")
        spec.input("optimized_structures_group_pk", valid_type=Int, help="PK of the group to store/accumulate optimized structures")
        spec.input("rslt_data", valid_type=PandasFrameData, help="result data in Pandas DataFrame format")
        spec.input("cryspy_in", valid_type=RinData, help="RinData for cryspy input")
        spec.input("detail_data", valid_type=(Dict, EAData), help="EA data for optimization")
        spec.input("id_queueing", valid_type=List, help="list of IDs for queuing structures for optimization")
        spec.input("code", valid_type=Code, help="label of your code")
        spec.input("potential", valid_type=ModelData, required=False, help="MLIP model data")
        spec.input("parameters", valid_type=Dict, help="calculation parameters")
        spec.input("options", valid_type=Dict, default=Dict, help="metadata.options")

        spec.output("structure_energy_data", valid_type=Dict, help="sorted energy results with structure data")
        spec.output("rslt_data", valid_type=PandasFrameData, help="result data in Pandas DataFrame format")
        spec.output_namespace("structure", valid_type=StructureData, dynamic=True)

        spec.exit_code(300, "ERROR_SUB_PROCESS_FAILED", message="One or more subprocesses failed.")


        spec.outline(
            cls.setup,
            while_(cls.should_run_batch)(
                cls.submit_batch,
                cls.process_batch_results,
            ),
            cls.collect_results # 最後に結果をまとめる
        )


    def setup(self):
        """
        最初に一度だけ呼ばれ、全体のタスクリストとバッチサイズを準備する。
        """
        self.ctx.ids_to_process = list(self.inputs.id_queueing)
        self.ctx.batch_size = 100  # <-- バッチサイズをここで設定
        self.ctx.all_submitted_calcs = {} # 全ての計算結果を保存する辞書


    # ★ whileループの継続条件メソッドを追加
    def should_run_batch(self):
        """
        処理すべきIDが残っていればTrueを返す。
        """
        return len(self.ctx.ids_to_process) > 0


    def submit_batch(self):

        group_pk = self.inputs.initial_structures_group_pk.value
        input_group = load_group(pk=group_pk)
        current_batch_ids = self.ctx.ids_to_process[:self.ctx.batch_size]

        # Groupから必要なNodeを探すためのマップを作る
        structure_map = {}
        for node in input_group.nodes:
            cid = node.base.extras.get('cryspy_id')
            if cid in current_batch_ids:
                structure_map[cid] = node

        self.report(f"Submitting optimization for {len(structure_map)} structures.")

        for cid,structure_node in structure_map.items():
            structure_node.store()
            self.out(f"structure.{cid}", structure_node)

            future = self.submit(optimization_WorkChain,
                code=self.inputs.code,
                structure=structure_node,
                parameters=self.inputs.parameters,
                options=self.inputs.options,
            )

            future.label = f"opt_{cid}"  # IDを文字列としてラベル付け

            self.to_context(calculations=append_(future))

        #処理した分を待ち行列から削除
        self.ctx.ids_to_process = self.ctx.ids_to_process[self.ctx.batch_size:]




    # ★ バッチごとの結果を処理するメソッドを追加
    def process_batch_results(self):
        """
        完了したバッチの結果を一時的に保存する。
        """
        # 結果を回収
        for calculation in self.ctx.calculations:
            # label ("opt_10") をキーにして保存 (辞書なので上書きされても害はないが、無駄な処理になる)
            self.ctx.all_submitted_calcs[calculation.label] = calculation

        # 【修正】処理が終わったらリストを空にする (次のバッチのためにリセット)
        self.ctx.calculations = []



    def collect_results(self):

        # RinDataと世代(gen)の取得（グループ名に使用するため先に取得）
        rin_data = self.inputs.cryspy_in
        rin = rin_data.rin

        gen = 1 # デフォルト値 (RSの場合など)
        if rin.algo == "EA":
            # EADataの場合はリストから世代を取得
            if isinstance(self.inputs.detail_data, Dict):
                 gen = self.inputs.detail_data.get_dict().get("ea_data")[0]
            else:
                 gen = self.inputs.detail_data.ea_data[0]



        # 初期構造辞書を復元
        input_group = load_group(pk=self.inputs.initial_structures_group_pk.value)
        init_struc_data = {}
        for node in input_group.nodes:
            cid = node.base.extras.get('cryspy_id')
            if cid is not None:
                init_struc_data[cid] = node.get_pymatgen()


        # 全世代の最適化後の構造を辞書に復元
        output_group = load_group(pk=self.inputs.optimized_structures_group_pk.value)
        opt_struc_data = {}
        for node in output_group.nodes:
            # extrasからIDを取得
            cid = node.base.extras.get('cryspy_id')
            if cid is not None:
                # pymatgenオブジェクトに変換して辞書に格納
                opt_struc_data[cid] = node.get_pymatgen()

        calcfunc_inputs = {}

        rslt_data_node = self.inputs.rslt_data
        rslt_data = rslt_data_node.df



        # 圧力設定の取得 (存在しなければ 0.0 GPa とする)
        target_pressure_gpa = 0.0
        try:
            optimizer_params = self.inputs.parameters.get_dict().get('optimizer', {})
            setup_params = optimizer_params.get('setup', {})
            # キーが存在しない、または None の場合は 0.0 を採用
            target_pressure_gpa = setup_params.get('scalar_pressure', 0.0)
            if target_pressure_gpa is None:
                target_pressure_gpa = 0.0
        except Exception:
            # 読み込みに失敗した場合も0.0 とする
            self.report("Warning: Could not read scalar_pressure. Assuming 0.0 GPa.")
            target_pressure_gpa = 0.0

        #self.report(f"Collecting results using Target Pressure = {target_pressure_gpa} GPa")



        # 4. 結果回収ループ
        for label, results_node in self.ctx.all_submitted_calcs.items():
            if not results_node.is_finished_ok:
                self.report(f'Sub-process {label} failed with exit status {results_node.exit_status}')
                continue

            cid_str = label.split('_')[-1]
            cid = int(cid_str)

            calcfunc_inputs[f"parameters_{cid_str}"] = results_node.outputs.parameters
            calcfunc_inputs[f"structure_{cid_str}"] = results_node.outputs.structure

            # Total Energy [eV]
            energy = results_node.outputs.parameters['total_energy']

            # Structure & Volume
            structure_node = results_node.outputs.structure
            opt_struc = structure_node.get_pymatgen()
            volume = opt_struc.volume       # [A^3]
            num_atoms = opt_struc.num_sites # [atoms]

            # H = E + PV
            # P=0なら pv_term=0 となり、H=E となる
            pv_term = (target_pressure_gpa * GPa) * volume
            enthalpy_total = energy + pv_term

            # 一原子あたりの値
            final_val_per_atom = enthalpy_total / num_atoms

            # ログ出力（デバッグ用）
            # P=0 のときは PV=0 と表示
            self.report(f"ID={cid}: E={energy:.2f}, P={target_pressure_gpa}GPa, PV={pv_term:.2f} -> H_total={enthalpy_total:.2f}")


            # print(f"ID: {cid}, Energy: {energy}")

            # Groupに追加
            structure_node.base.extras.set('cryspy_id', cid)
            output_group.add_nodes(structure_node)
            #self.report(f"Added StructureData<{structure_node.pk}> with cryspy_id={cid} to Group<{output_group.pk}>")

            gen_arg = None
            if rin.algo == "EA":
                gen_arg = gen

            try:
                # CrySPY登録
                opt_struc_data, rslt_data = ctrl_job.regist_opt(
                    rin,
                    cid,
                    init_struc_data,
                    opt_struc_data,
                    rslt_data,
                    opt_struc,
                    final_val_per_atom,
                    magmom=None,
                    check_opt=None,
                    ef=None,
                    nat=None,
                    n_selection=None,
                    gen=gen_arg
                )
            except Exception as e:
                self.report(f"ERROR: Failed to register structure ID: {cid}. Skipping this structure.")
                self.report(f"Reason: {e}")
                continue

        if calcfunc_inputs:
            structure_energy_data_results = pack_results(**calcfunc_inputs)
            self.out("structure_energy_data", structure_energy_data_results)

        rslt_node = PandasFrameData(rslt_data)
        rslt_node.store()
        self.out('rslt_data', rslt_node)

        self.report(f"Generation {gen} All structures optimization Done.")





