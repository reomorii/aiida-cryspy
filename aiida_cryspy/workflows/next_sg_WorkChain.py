from aiida.orm import List,Int,load_group
from aiida.engine import WorkChain,calcfunction
from aiida.plugins import DataFactory
from cryspy.job import ctrl_job

StructureCollectionData = DataFactory("aiida_cryspy.structurecollection")
PandasFrameData = DataFactory("aiida_cryspy.dataframe")
RinData = DataFactory("aiida_cryspy.rin_data")
EAData = DataFactory("aiida_cryspy.ea_data")
StructureData = DataFactory("core.structure")

@calcfunction
def next_sg_gen_cryspy(**kwargs):

    rin = kwargs["cryspy_in"].rin
    detail_data = kwargs["detail_data"]
    print("\n--- Contents of detail_data (input) ---")
    # detail_data は EAData 型なので、その中の ea_data 属性を表示します
    print(detail_data.ea_data)
    print("----------------------------------------\n")
    # --------------------------------
    gen = kwargs["detail_data"].ea_data[0]
    init_struc_data = kwargs["initial_structures"].structurecollection
    rslt_data = kwargs["rslt_data"].df
    group_pk = kwargs["structures_group_pk"].value

    go_next_sg = True
    nat_data = None #組成可変のもの
    structure_mol_id = None


    # Groupから最適化済み構造を読み込み、辞書を再構築する
    group = load_group(pk=group_pk)
    opt_struc_data = {}
    for node in group.nodes:
        # extraに保存したIDを取得
        cid = node.base.extras.get("cryspy_id")
        # cidをキーとしてpymatgenオブジェクトを辞書に格納
        opt_struc_data[cid] = node.get_pymatgen()

    # --- opt_struc_data の中身を表示 ---
    print("\n--- Contents of opt_struc_data ---")
    # 辞書の各要素（IDと構造）をループで表示します
    for cryspy_id, structure in opt_struc_data.items():
        print(f"  ID: {cryspy_id}")
        # structureはpymatgenオブジェクトなので、そのままprintすると要約情報が表示されます
        print(f"  Structure: {structure}")
        print("-" * 20)
    print("----------------------------------\n")
    # ------------------------------------


    # import os
    # print(f"Current working directory next_sg: {os.getcwd()}") # 現在のディレクトリを確認

    # print(f"--- Inside calcfunction ---")
    # print(f"os.getcwd() reports: {os.getcwd()}")
    # print(f"Actual contents of this directory: {os.listdir('.')}") # この行を追加
    # print(f"--------------------------")

    init_struc, id_queueing, ea_data_node, rslt_data = ctrl_job.next_gen_EA(
        rin,
        gen,
        go_next_sg,
        init_struc_data,
        opt_struc_data,
        rslt_data,
        nat_data,
        structure_mol_id
    )

    return {
        "next_structures": StructureCollectionData(structures=init_struc),
        "id_queueing": List(list=id_queueing),
        "detail_data": EAData(ea_data=ea_data_node),
        "rslt_data": PandasFrameData(rslt_data),
    }


class next_sg_WorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("initial_structures", valid_type=StructureCollectionData)
        #spec.input("opt_structures", valid_type=StructureCollectionData)
        spec.input("rslt_data", valid_type=PandasFrameData)
        spec.input("detail_data", valid_type=EAData)
        spec.input("cryspy_in", valid_type=RinData, help='cryspy input data')
        spec.input("structures_group_pk", valid_type=Int, help='PK of the group with optimized structures.')

        spec.output("next_structures", valid_type=StructureCollectionData, help='next generation structures')
        spec.output("rslt_data",valid_type=PandasFrameData, help='result data in Pandas DataFrame format')
        spec.output("detail_data", valid_type=EAData, help='evolutionary algorithm data for next generation')
        spec.output("id_queueing", valid_type=List, help='queueing ids for next generation')


        spec.outline(
            cls.call_next_sg,
            cls.set_outputs
        )

    def call_next_sg(self):

        # rin_data = self.inputs.cryspy_in        # rin オブジェクトを取り出す
        # rin = rin_data.rin  # ← Python オブジェクトとして使用可能

        # gen = self.inputs.detail_data.ea_data[0]  # gen（世代）を取得

        # init_struc_data = self.inputs.initial_structures.structurecollection
        # #opt_struc_data = self.inputs.opt_structures.structurecollection
        # rslt_data = self.inputs.rslt_data.df

        # group_pk = self.inputs.structures_group_pk.value

        # inputs = {
        #             'rin': rin,
        #             'gen': gen,
        #             'initial_structures': init_struc_data,
        #             'rslt_data': rslt_data,
        #             'structures_group_pk': group_pk,
        #         }

        # self.report(f"Reconstructed {len(opt_struc_data)} optimized structures from Group<{group.pk}>.")
        # import time
        # self.report("Pausing for 50 seconds to ensure file system consistency...")
        # time.sleep(50)
        # self.report("Wait finished, proceeding with next generation.")

        # import os
        # self.report("--- START: Listing contents of the current working directory ---")

        # # 現在の作業ディレクトリのパスを取得
        # cwd = os.getcwd()
        # self.report(f"Current Path: {cwd}")

        # # os.walkを使って、カレントディレクトリ以下の全ファイルと全フォルダをリストアップ
        # file_list = []
        # for root, dirs, files in os.walk(cwd):
        #     # 現在調べているフォルダのパスを表示
        #     path = root.split(os.sep)
        #     self.report(f"Directory: {os.path.join(*path)}")

        #     # そのフォルダ内にあるサブフォルダの一覧を表示
        #     for d in dirs:
        #         self.report(f"  Sub-directory: {d}")

        #     # そのフォルダ内にあるファイルの一覧を表示
        #     for f in files:
        #         self.report(f"  File: {f}")
        # #         file_list.append(os.path.join(root, f))

        # if not file_list:
        #     self.report("!!! WARNING: The working directory is EMPTY. No files or folders found. !!!")

        # self.report("--- END: Directory listing complete ---")

        inputs = self.inputs
        results = next_sg_gen_cryspy(**inputs)
        self.ctx.results = results
        self.report("next_sg_gen_cryspy() has been called to generate next generation structures.")

        # self.out("next_structures", StructureCollectionData(structures=results["next_structures"]))
        # self.out("id_queueing", List(list=results["id_queueing"]))
        # self.out("detail_data", EAData(ea_data=results["detail_data"]))
        # self.out("rslt_data", PandasFrameData(df=results["rslt_data"])))

        # next_struc_node = StructureCollectionData(structures=results["next_structures"])
        # next_struc_node.store()
        # self.out("next_structures", next_struc_node)

        # id_queueing_node = List(list=results["id_queueing"])
        # id_queueing_node.store()
        # self.out("id_queueing", id_queueing_node)

        # detail_data_node = EAData(ea_data=results["detail_data"])
        # detail_data_node.store()
        # self.out("detail_data", detail_data_node)

        # rslt_node = PandasFrameData(results["rslt_data"])
        # rslt_node.store()
        # self.out("rslt_data", rslt_node)


    def set_outputs(self):

        # WorkChainの出力に結果を接続
        self.out("next_structures", self.ctx.results["next_structures"])
        self.out("rslt_data", self.ctx.results["rslt_data"])
        self.out("detail_data", self.ctx.results["detail_data"])
        self.out("id_queueing", self.ctx.results["id_queueing"])

