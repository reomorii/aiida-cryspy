from aiida.orm import Dict,Str,List,Int,Group
from aiida.engine import WorkChain,ToContext,calcfunction
from aiida.plugins import DataFactory
from cryspy.start import cryspy_init
import os

StructureCollectionData = DataFactory("aiida_cryspy.structurecollection")
PandasFrameData = DataFactory("aiida_cryspy.dataframe")
RinData = DataFactory("aiida_cryspy.rin_data")
EAData = DataFactory("aiida_cryspy.ea_data")
StructureData = DataFactory("core.structure")


class initialize_workchain(WorkChain):

    @classmethod
    def define(cls,spec):
        super().define(spec)
        spec.input("cryspy_in_filename", valid_type=Str)

        spec.output("initial_structures_group_pk", valid_type=Int)
        spec.output("optimized_structures_group_pk", valid_type=Int)
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

        # print(f"Current working directory init: {os.getcwd()}") # 現在のディレクトリを確認
 
        init_struc_data, _, rin, rslt_data, detail_data, id_queueing = cryspy_init.initialize()

        # グループの作成
        group_label = f"cryspy_gen_1_init_{self.uuid}"
        group = Group(label=group_label)
        group.store()

        # 構造を1つずつ保存してGroupに入れる
        for cid, pmg_struct in init_struc_data.items():
            s_node = StructureData(pymatgen=pmg_struct)
            s_node.base.extras.set('cryspy_id', cid) # IDを付与
            s_node.store()
            group.add_nodes(s_node)

        self.report(f"Stored {len(init_struc_data)} structures to Group<{group.pk}>.")
        
        optimized_group_label = f"cryspy_optimized_{self.uuid}"
        optimized_group = Group(label=optimized_group_label)
        optimized_group.store()

        # 結果をContextに保存
        self.ctx._init_group_pk = group.pk
        self.ctx._optimized_group_pk = optimized_group.pk
        self.ctx.rin = rin
        self.ctx.rslt_data = rslt_data
        self.ctx.detail_data = detail_data
        self.ctx.id_queueing = id_queueing

    def set_outputs_and_cleanup(self):
        """
        出力を設定
        """

        group_pk_node = Int(self.ctx._init_group_pk)
        group_pk_node.store()
        self.out("initial_structures_group_pk", group_pk_node)

        optimized_group_pk_node = Int(self.ctx._optimized_group_pk)
        optimized_group_pk_node.store()
        self.out("optimized_structures_group_pk", optimized_group_pk_node)

        # RinData
        rin_data_node = RinData(self.ctx.rin)
        rin_data_node.store()
        self.out("cryspy_in", rin_data_node)

        # Result Data
        rslt_data_node = PandasFrameData(self.ctx.rslt_data)
        rslt_data_node.store()
        self.out("rslt_data", rslt_data_node)

        # Detail Data
        if self.ctx.rin.algo == "EA":
            detail_data_node = EAData(self.ctx.detail_data)
        else:
            detail_data_node = Dict(dict=self.ctx.detail_data)
        detail_data_node.store()
        self.out("detail_data", detail_data_node)

        # ID Queueing
        id_queueing_node = List(list=self.ctx.id_queueing)
        id_queueing_node.store()
        self.out("id_queueing", id_queueing_node)

        # lock_cryspyの削除 (action.initialize()の末尾部分を再現)
        if os.path.isfile("lock_cryspy"):
            os.remove("lock_cryspy")
            self.report("Removed lock_cryspy file.")

