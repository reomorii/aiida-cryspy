from aiida.orm import Dict,Str,List
from aiida.engine import WorkChain,ToContext,calcfunction
from aiida.plugins import DataFactory
from cryspy.start import cryspy_init
import os

StructureCollectionData = DataFactory("aiida_cryspy.structurecollection")
PandasFrameData = DataFactory("aiida_cryspy.dataframe")
RinData = DataFactory("aiida_cryspy.rin_data")
EAData = DataFactory("aiida_cryspy.ea_data")
StructureData = DataFactory("core.structure")


@calcfunction
def initialize_cryspy_data():

    import os
    print(f"Current working directory next_sg: {os.getcwd()}") # 現在のディレクトリを確認
    init_struc_data, _, rin, rslt_data, detail_data, id_queueing = cryspy_init.initialize()

    return {
        "initial_structures": StructureCollectionData(structures=init_struc_data),
        "cryspy_in": RinData(rin),
        "rslt_data": PandasFrameData(rslt_data),
        "detail_data": EAData(detail_data) if rin.algo == "EA" else Dict(dict=detail_data),
        "id_queueing": List(list=id_queueing),
    }


class initialize_workchain(WorkChain):

    @classmethod
    def define(cls,spec):
        super().define(spec)
        spec.input("cryspy_in_filename", valid_type=Str)

        spec.output("initial_structures",valid_type=StructureCollectionData)
        #spec.output("opt_structures",valid_type=StructureCollectionData)
        spec.output("rslt_data", valid_type=PandasFrameData)
        spec.output("cryspy_in", valid_type=RinData)
        spec.output("detail_data", valid_type=(Dict, EAData))
        spec.output("id_queueing", valid_type=List)

        spec.exit_code(101, "ERROR_LOCK_FILE_EXISTS", message="lock_cryspy file already exists.")
        spec.exit_code(102, "ERROR_STAT_FILE_EXISTS", message="cryspy.stat file already exists.")

        spec.outline(
            cls.prepare_and_check,
            cls.run_initialize,
            cls.set_outputs_and_cleanup
        )

    def prepare_and_check(self):
        """
        ファイル操作と状態チェックを行う。
        """
        # lock_cryspyのチェックと作成 (action.initialize()の冒頭部分を再現)
        if os.path.isfile("lock_cryspy"):
            self.report("lock_cryspy file exists, aborting.")
            return self.exit_codes.ERROR_LOCK_FILE_EXISTS  # (別途exit_codeの定義が必要)
        with open("lock_cryspy", "w"):
            pass
        self.report("Created lock_cryspy file.")

        # cryspy.statのチェック
        if os.path.isfile("cryspy.stat"):
            self.report("cryspy.stat file exists. Clean files to start from the beginning.")
            return self.exit_codes.ERROR_STAT_FILE_EXISTS  # (別途exit_codeの定義が必要)

    def run_initialize(self):
        """
        純粋なデータ生成処理をcalcfunctionとして実行する。
        """
        # cryspy_init.initialize()を直接呼び出す
        self.report("Running cryspy_init.initialize() data generation.")

        # calcfunctionを呼び出して結果をAiiDAノードに変換

        print(f"Current working directory init: {os.getcwd()}") # 現在のディレクトリを確認
        results = initialize_cryspy_data()
        self.ctx.results = results
        self.report("Data generation finished and converted to AiiDA nodes.")

    def set_outputs_and_cleanup(self):
        """
        出力を設定し、後片付けを行う。
        """
        # WorkChainの出力に結果を接続
        self.out("initial_structures", self.ctx.results["initial_structures"])
        self.out("cryspy_in", self.ctx.results["cryspy_in"])
        self.out("rslt_data", self.ctx.results["rslt_data"])
        self.out("detail_data", self.ctx.results["detail_data"])
        self.out("id_queueing", self.ctx.results["id_queueing"])

        # lock_cryspyの削除 (action.initialize()の末尾部分を再現)
        if os.path.isfile("lock_cryspy"):
            os.remove("lock_cryspy")
            self.report("Removed lock_cryspy file.")

