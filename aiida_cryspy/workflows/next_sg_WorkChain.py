from aiida.orm import List,Int,load_group,Group
from aiida.engine import WorkChain,calcfunction
from aiida.plugins import DataFactory
from cryspy.job import ctrl_job

StructureCollectionData = DataFactory("aiida_cryspy.structurecollection")
PandasFrameData = DataFactory("aiida_cryspy.dataframe")
RinData = DataFactory("aiida_cryspy.rin_data")
EAData = DataFactory("aiida_cryspy.ea_data")
StructureData = DataFactory("core.structure")


class next_sg_WorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # spec.input("initial_structures", valid_type=StructureCollectionData)
        spec.input("initial_structures_group_pk", valid_type=Int, help="PK of the group containing optimized structures from previous generation")
        spec.input("optimized_structures_group_pk", valid_type=Int, help="PK of the group containing OPTIMIZED structures (Post-optimization)")
        #spec.input("opt_structures", valid_type=StructureCollectionData)
        spec.input("rslt_data", valid_type=PandasFrameData)
        spec.input("detail_data", valid_type=EAData)
        spec.input("cryspy_in", valid_type=RinData, help='cryspy input data')
        # spec.input("structures_group_pk", valid_type=Int, help='PK of the group with optimized structures.')

        # spec.output("next_structures", valid_type=StructureCollectionData, help='next generation structures')
        spec.output("next_structures_group_pk", valid_type=Int, help='PK of the group containing next generation structures')
        spec.output("rslt_data",valid_type=PandasFrameData, help='result data in Pandas DataFrame format')
        spec.output("detail_data", valid_type=EAData, help='evolutionary algorithm data for next generation')
        spec.output("id_queueing", valid_type=List, help='queueing ids for next generation')


        spec.outline(
            cls.call_next_sg,
            cls.set_outputs
        )

    def call_next_sg(self):



        self.report("Starting next structure generation...")


        # 1. 最適化「前」の構造データを復元 (init_struc_data)
        init_group_pk = self.inputs.initial_structures_group_pk.value
        init_group = load_group(pk=init_group_pk)

        init_struc_data = {}
        for node in init_group.nodes:
            cid = node.base.extras.get("cryspy_id")
            if cid is not None:
                init_struc_data[cid] = node.get_pymatgen()

        # 2. 最適化「後」の構造データを復元 (opt_struc_data)
        opt_group_pk = self.inputs.optimized_structures_group_pk.value
        opt_group = load_group(pk=opt_group_pk)

        opt_struc_data = {}
        for node in opt_group.nodes:
            cid = node.base.extras.get("cryspy_id")
            if cid is not None:
                opt_struc_data[cid] = node.get_pymatgen()



        rin = self.inputs.cryspy_in.rin
        gen = self.inputs.detail_data.ea_data[0]
        rslt_data = self.inputs.rslt_data.df
        go_next_sg = True
        nat_data = None #組成可変のもの
        structure_mol_id = None

        self.report(f"Generating generation {gen + 1} from {len(opt_struc_data)} parent structures.")

        # 2. 次世代生成ロジックの実行
        # ctrl_job.next_gen_EA を直接呼び出します
        # (calcfunctionにすると戻り値の構造辞書が巨大になりDBエラーになるため)
        next_struc_dict, id_queueing, ea_data, rslt_data_new = ctrl_job.next_gen_EA(
            rin,
            gen,
            go_next_sg,
            init_struc_data,
            opt_struc_data,
            rslt_data,
            nat_data,
            structure_mol_id
        )

        # 3. 新しいGroupの作成と保存
        # 次世代の番号
        next_gen = gen + 1
        new_group_label = f"cryspy_gen_{next_gen}_init_{self.uuid}"
        output_group = Group(label=new_group_label)
        output_group.store()

        self.report(f"Storing {len(next_struc_dict)} next generation structures to Group<{output_group.pk}>")

        # 構造を保存してGroupに追加
        for cid, pmg_struct in next_struc_dict.items():
            # AiiDAのStructureDataに変換
            s_node = StructureData(pymatgen=pmg_struct)
            # CrySPY ID を extra に付与
            s_node.base.extras.set('cryspy_id', cid)
            # 保存
            s_node.store()
            # グループに追加
            output_group.add_nodes(s_node)

        # 6. コンテキストに保存
        self.ctx.next_group_pk = output_group.pk
        self.ctx.rslt_data = rslt_data_new
        self.ctx.detail_data = ea_data
        self.ctx.id_queueing = id_queueing

        self.report(f"Next generation (Gen {next_gen}) creation finished.")

    def set_outputs(self):
        # 出力設定
        next_group_pk_node = Int(self.ctx.next_group_pk)
        next_group_pk_node.store()
        self.out("next_structures_group_pk", next_group_pk_node)

        rslt_data_node = PandasFrameData(self.ctx.rslt_data)
        rslt_data_node.store()
        self.out("rslt_data", rslt_data_node)

        detail_data_node = EAData(ea_data=self.ctx.detail_data)
        detail_data_node.store()
        self.out("detail_data", detail_data_node)

        id_queueing_node = List(list=self.ctx.id_queueing)
        id_queueing_node.store()
        self.out("id_queueing", id_queueing_node)
