from aiida.orm import List
from aiida.engine import WorkChain
from aiida.plugins import DataFactory
from cryspy.job import ctrl_job

StructureCollectionData = DataFactory("aiida_cryspy.structurecollection")
PandasFrameData = DataFactory('aiida_cryspy.dataframe')
RinData = DataFactory('aiida_cryspy.rin_data')
EAData = DataFactory('aiida_cryspy.ea_data')
StructureData = DataFactory('core.structure')


class next_sg_WorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("initial_structures", valid_type=StructureCollectionData)
        spec.input("opt_structures", valid_type=StructureCollectionData)
        spec.input("rslt_data", valid_type=PandasFrameData)
        spec.input("detail_data", valid_type=EAData)
        spec.input("cryspy_in", valid_type=RinData, help='cryspy input data')

        spec.output("next_structures", valid_type=StructureCollectionData, help='next generation structures')
        spec.output("rslt_data",valid_type=PandasFrameData, help='result data in Pandas DataFrame format')
        spec.output("detail_data", valid_type=EAData, help='evolutionary algorithm data for next generation')
        spec.output("id_queueing", valid_type=List, help='queueing ids for next generation')


        spec.outline(
            cls.call_next_sg
        )

    def call_next_sg(self):

        rin_data = self.inputs.cryspy_in        # rin オブジェクトを取り出す
        rin = rin_data.rin  # ← Python オブジェクトとして使用可能

        gen = self.inputs.detail_data.ea_data[0]  # gen（世代）を取得

        go_next_sg = True

        init_struc_data = self.inputs.initial_structures.structurecollection
        opt_struc_data = self.inputs.opt_structures.structurecollection

        rslt_data = self.inputs.rslt_data.df

        nat_data = None
        structure_mol_id = None

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

        init_struc_node = StructureCollectionData(structures=init_struc)
        init_struc_node.store()
        self.out('next_structures', init_struc_node)

        id_queueing_node = List(list=id_queueing)
        id_queueing_node.store()
        self.out("id_queueing", id_queueing_node)

        detail_data_node = EAData(ea_data=ea_data_node)
        detail_data_node.store()
        self.out("detail_data", detail_data_node)

        rslt_node = PandasFrameData(rslt_data)
        rslt_node.store()
        self.out('rslt_data', rslt_node)