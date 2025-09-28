from aiida.orm import Dict,List,Code,ArrayData,RemoteData,FolderData
from aiida.engine import WorkChain,calcfunction,ToContext,while_
from aiida.plugins import DataFactory
import os

from cryspy.job import ctrl_job
from aiida_mlip.data.model import ModelData

StructureCollectionData = DataFactory("aiida_cryspy.structurecollection")
PandasFrameData = DataFactory('aiida_cryspy.dataframe')
RinData = DataFactory('aiida_cryspy.rin_data')
EAData = DataFactory('aiida_cryspy.ea_data')
StructureData = DataFactory('core.structure')


class optimization_WorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=Code, help='label of your code')
        spec.input("structure", valid_type=StructureData, help='selected structure for optimization')
        spec.input('parameters', valid_type=Dict)
        spec.input('options', valid_type=Dict, default=Dict, help='metadata.options')

        spec.output("remote_folder",valid_type=RemoteData, help='remote folder of the workchain')
        spec.output("retrieved", valid_type=FolderData, help='retrieved data from the workchain')
        spec.output("structure", valid_type=StructureData, help='optimized structure from the workchain')
        spec.output("array", valid_type=ArrayData, help='array data from the workchain')
        spec.output("parameters", valid_type=Dict, help='output parameters from the workchain')

        spec.outline(
            cls.submit_workchains,
            cls.inspect_workchains
        )


        # spec.output("retrieved", valid_type=FolderData)
        # spec.output("opt_structure", valid_type=StructureData)



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
        builder.metadata.options.parser_name = 'ase.ase'
        builder.metadata.options.additional_retrieve_list = ['opt.traj', 'opt_struc.vasp']
        # submit workchain
        future = self.submit(builder)
        return ToContext(my_future=future)


    def inspect_workchains(self):
        #sleepを入れて並列を確認
        calculations = self.ctx.my_future

        if 'remote_folder' in calculations.outputs:
            self.out("remote_folder", calculations.outputs.remote_folder)
        if 'array' in calculations.outputs:
            self.out("array", calculations.outputs.array)
        if 'retrieved' in calculations.outputs:
            self.out("retrieved", calculations.outputs.retrieved)
        if 'parameters' in calculations.outputs:
            self.out("parameters", calculations.outputs.parameters)
        if 'structure' in calculations.outputs:
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
        if key.startswith('parameters_'):
            grouped_data[id_]['parameters'] = node
        elif key.startswith('structure_'):
            grouped_data[id_]['structure'] = node

    # グループ化されたデータを処理
    for id_, data in grouped_data.items():
        # parametersノードからエネルギーを取得
        energy = data['parameters'].get_dict().get('total_energy')

        # structureノードからpymatgenオブジェクトを取得し、JSON互換の辞書に変換
        struc_dict = data['structure'].get_pymatgen().as_dict()

        final_results[id_] = {'energy': energy, 'structure': struc_dict}

    # Python辞書をAiiDAのDictノードとして返す
    return Dict(dict=final_results)


class multi_structure_optimize_WorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("initial_structures", valid_type=StructureCollectionData, help='initial structure data for optimization')
        spec.input("opt_structures", valid_type=StructureCollectionData, help='optimized structure data')
        spec.input("rslt_data", valid_type=PandasFrameData, help='result data in Pandas DataFrame format')
        spec.input("cryspy_in", valid_type=RinData, help='RinData for cryspy input')
        spec.input("detail_data",valid_type=(Dict,EAData), help='EA data for optimization')
        spec.input("id_queueing", valid_type=List, help='list of IDs for queuing structures for optimization')
        spec.input("code", valid_type=Code, help='label of your code')
        spec.input("potential", valid_type=ModelData, required=False, help='MLIP model data')
        spec.input("parameters", valid_type=Dict, help='calculation parameters')
        spec.input("options", valid_type=Dict, default=Dict, help='metadata.options')

        spec.output("structure_energy_data", valid_type=Dict, help='sorted energy results with structure data')
        spec.output("opt_struc_data", valid_type=StructureCollectionData, help='optimized structure data')
        spec.output("rslt_data", valid_type=PandasFrameData, help='result data in Pandas DataFrame format')
        spec.output_namespace("structure", valid_type=StructureData, dynamic=True)

        spec.exit_code(300, 'ERROR_SUB_PROCESS_FAILED', message='One or more subprocesses failed.')

        # spec.outline(
        #     cls.optimize,
        #     cls.collect_results
        # )

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
        self.ctx.batch_size = 2  # <-- バッチサイズをここで設定
        self.ctx.all_submitted_calcs = {} # 全ての計算結果を保存する辞書


    # ★ whileループの継続条件メソッドを追加
    def should_run_batch(self):
        """
        処理すべきIDが残っていればTrueを返す。
        """
        return len(self.ctx.ids_to_process) > 0


    def submit_batch(self):
        initial_structures_dict = self.inputs.initial_structures.structurecollection
        calculations = {}

        ids_this_batch = self.ctx.ids_to_process[:self.ctx.batch_size]

        for id in ids_this_batch:
            structure_ = initial_structures_dict[id]
            structure = StructureData(pymatgen=structure_)
            structure.store()
            self.out(f"structure.{id}", structure)

            future = self.submit(optimization_WorkChain,
                code=self.inputs.code,
                structure=structure,
                parameters=self.inputs.parameters,
                options=self.inputs.options,
            )

            label = f"opt_{id}"  # IDを文字列としてラベル付け
            calculations[label] = future

        # 処理が終わったIDを全体のリストから削除
        self.ctx.ids_to_process = self.ctx.ids_to_process[self.ctx.batch_size:]
        self.report(f"Submitted a batch of {len(calculations)} calculations. "
                            f"{len(self.ctx.ids_to_process)} calculations remaining.")

        return ToContext(**calculations)



    # ★ バッチごとの結果を処理するメソッドを追加
    def process_batch_results(self):
        """
        完了したバッチの結果を一時的に保存する。
        """
        finished_batch = {key: self.ctx[key] for key in self.ctx if key.startswith('opt_')}
        self.ctx.all_submitted_calcs.update(finished_batch)



    def collect_results(self):
        init_struc_data = self.inputs.initial_structures.structurecollection
        opt_struc_data = self.inputs.opt_structures.structurecollection
        calcfunc_inputs = {}

        # ---------- mkdir work/fin
        # os.makedirs('work/fin', exist_ok=True)

        rslt_data_node = self.inputs.rslt_data
        # pandas.DataFrame として取り出す
        rslt_data = rslt_data_node.df


        #     # 成功判定のチェック
        # for id, calculation in self.ctx.items():
        #     if not calculation.is_finished_ok:
        #         self.report(f'Sub-process for ID {id} failed with exit status {calculation.exit_status}')
        #         return self.exit_codes.ERROR_SUB_PROCESS_FAILED

            # 成功判定のチェック
        for cid, results_node in self.ctx.all_submitted_calcs.items():
            if not results_node.is_finished_ok:
                self.report(f'Sub-process for ID {cid} failed with exit status {results_node.exit_status}')
                continue
                # return self.exit_codes.ERROR_SUB_PROCESS_FAILED

            calcfunc_inputs[f"parameters_{cid}"] = results_node.outputs.parameters
            calcfunc_inputs[f"structure_{cid}"] = results_node.outputs.structure

            rin_data = self.inputs.cryspy_in        # rin オブジェクトを取り出す
            rin = rin_data.rin                      # ← Python オブジェクトとして使用可能

            energy = results_node.outputs.parameters['total_energy']  # 'total_energy' キーからエネルギーを取得
            opt_struc = results_node.outputs.structure.get_pymatgen()

            print("energy:", energy)

            cid = int(cid.split('_')[-1])  # IDを整数に変換
            # os.makedirs(f'work/{cid}',exist_ok=True)
            # work_path = f'work/{cid}/'

            if rin.algo == "RS":
                gen_ = None

            elif rin.algo == "EA":
                gen_ = self.inputs.detail_data.ea_data[0]  # EADataから世代情報を取得

            #cryspyによる結果の保存
            opt_struc_data, rslt_data = ctrl_job.regist_opt(
                rin,
                cid,
                init_struc_data,
                opt_struc_data,
                rslt_data,
                opt_struc,
                energy,
                magmom=None,
                check_opt=None,
                ef=None,
                nat=None,
                n_selection=None,
                gen=gen_
            )

        # 成功した計算が一つでもあれば、後続の処理を実行
        if calcfunc_inputs:
            structure_energy_data_results = pack_results(**calcfunc_inputs)
            self.out("structure_energy_data", structure_energy_data_results)

        opt_struc_node = StructureCollectionData(structures=opt_struc_data)
        opt_struc_node.store()
        self.out('opt_struc_data', opt_struc_node)

        rslt_node = PandasFrameData(rslt_data)
        rslt_node.store()
        self.out('rslt_data', rslt_node)