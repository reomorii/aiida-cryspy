from aiida.orm import Dict,Str,List
from aiida.engine import WorkChain
from aiida.plugins import DataFactory
from cryspy.interactive import action


StructureCollectionData = DataFactory("aiida_cryspy.structurecollection")
PandasFrameData = DataFactory('aiida_cryspy.dataframe')
RinData = DataFactory('aiida_cryspy.rin_data')
EAData = DataFactory('aiida_cryspy.ea_data')
StructureData = DataFactory('core.structure')


class initialize_workchain(WorkChain):

    @classmethod
    def define(cls,spec):
        super().define(spec)
        spec.input("cryspy_in_filename", valid_type=Str)

        spec.output("initial_structures",valid_type=StructureCollectionData)
        spec.output("opt_structures",valid_type=StructureCollectionData)
        spec.output("rslt_data",valid_type=PandasFrameData)
        spec.output("cryspy_in", valid_type=RinData)
        spec.output('detail_data', valid_type=(Dict, EAData))
        spec.output("id_queueing",valid_type=List)


        spec.outline(
            cls.call_crsypy_initialize
        )

    def call_crsypy_initialize(self):

        init_struc_data, opt_struc_data, rin, rslt_data, detail_data, id_queueing = action.initialize()

        init_struc_node = StructureCollectionData(structures=init_struc_data)
        init_struc_node.store()
        self.out('initial_structures', init_struc_node)

        opt_struc_node = StructureCollectionData(structures=opt_struc_data)
        opt_struc_node.store()
        self.out('opt_structures', opt_struc_node)

        cryspy_in = RinData(rin)
        cryspy_in.store()
        self.out("cryspy_in", cryspy_in)

        rslt_node = PandasFrameData(rslt_data)
        rslt_node.store()
        self.out('rslt_data', rslt_node)

        id_queueing_node = List(list=id_queueing)
        id_queueing_node.store()
        self.out("id_queueing", id_queueing_node)

        algo = rin.algo

        if algo == "RS":
            RS_node = Dict(dict=detail_data)
            RS_node.store()
            self.out('detail_data', RS_node)

        elif algo == "EA":
            ea_node = EAData(detail_data)
            ea_node.store()
            self.out('detail_data', ea_node)

        else:
            raise ValueError(f'algo not supported. algo={algo}')