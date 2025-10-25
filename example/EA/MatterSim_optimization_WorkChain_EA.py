#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from aiida.orm import Dict,load_code,Int
from aiida.engine import run
from aiida.plugins import WorkflowFactory


# In[ ]:


from aiida import load_profile
load_profile()


# In[ ]:


initialize_WorkChain = WorkflowFactory('aiida_cryspy.initial_structures')
result,node = run.get_node(initialize_WorkChain, cryspy_in_filename="cryspy_in")


# In[ ]:


result


# In[ ]:


initial_structures_node_1 = result['initial_structures']
tmp_rslt_data_node = result['rslt_data']
cryspy_in_node = result['cryspy_in']
detail_data_node_1 = result['detail_data']
id_queueing_node_1 = result['id_queueing']


# In[ ]:


code = load_code("aiida-ase@script_ssh2")

prepend_commands = """
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=2
"""


parameters = Dict(dict={
        'calculator': {
                # 'name': 'gpaw',
                'args': {
                'load_path': '/home/morii22/.local/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth',
                'device': "cpu", # pre_linesで定義したdevice変数を参照
                },
        },
        'optimizer': {
                'name': 'BFGS',
                'run_args': {
                'fmax': 0.01,
                'steps': 2000
                },
                'setup':{
                'FixSymmetry':True,
                'FrechetCellFilter':True,
                },
        },
        'extra_imports': [
                ['mattersim.forcefield', 'MatterSimCalculator'],
                'torch',
        ],
        'pre_lines': [
                "custom_calculator = MatterSimCalculator",
                # "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        ],
        'post_lines': [
                'for key, value in results.items():',
                # 値が配列ではなく、かつ.item()メソッドを持つ場合のみ変換
                '    if not isinstance(value, numpy.ndarray) and hasattr(value, "item"):',
                '        results[key] = value.item()',
                'atoms.calc = None'
        ],
        'atoms_getters': [
                'potential_energy',
                'forces',
        ],
})

inputs = {
        'initial_structures':initial_structures_node_1,
        'rslt_data':tmp_rslt_data_node,
        'cryspy_in':cryspy_in_node,
        'detail_data':detail_data_node_1,
        'id_queueing':id_queueing_node_1,
        "code": code,
        "parameters": parameters,
        "options": Dict(dict={"resources": {'tot_num_mpiprocs': 4, 'parallel_env': 'smp'},
                              'max_wallclock_seconds': 600, 'prepend_text': prepend_commands,'queue_name': 'ibis2.q'}),
        "structures_group_pk": Int(5)
}

        # "options": Dict(dict={"resources": {'tot_num_mpiprocs': 4,
        #                                  'parallel_env': 'smp'},
        #                                  'max_wallclock_seconds': 600, 'prepend_text': prepend_commands})


# In[ ]:


optimize_WorkChain = WorkflowFactory('aiida_cryspy.optimize_structures')
result,node = run.get_node(optimize_WorkChain, **inputs)


# In[ ]:


result


# In[ ]:


initial_structures_node_1
rslt_data_node_1 = result['rslt_data']
detail_data_node_1


# In[ ]:


input = {
    'initial_structures': initial_structures_node_1,
    'rslt_data': result['rslt_data'],
    'detail_data': detail_data_node_1,
    'cryspy_in': cryspy_in_node,
    'structures_group_pk': Int(5),
}


# In[ ]:


next_sg_WorkChain = WorkflowFactory('aiida_cryspy.next_sg')
result, node = run.get_node(next_sg_WorkChain, **input)


# In[ ]:


result


# In[ ]:


initial_structures_node_2 = result['next_structures']
rslt_data_node_2 = result['rslt_data']
detail_data_node_2 = result['detail_data']
id_queueing_node_2 = result['id_queueing']


# In[ ]:


inputs = {
        'initial_structures':initial_structures_node_2,
        'rslt_data':rslt_data_node_2,
        'cryspy_in':cryspy_in_node,
        'detail_data':detail_data_node_2,
        'id_queueing':id_queueing_node_2,
        "code": code,
        "parameters": parameters,
        "options": Dict(dict={"resources": {'tot_num_mpiprocs': 4, 'parallel_env': 'smp'},
                              'max_wallclock_seconds': 600, 'prepend_text': prepend_commands, 'queue_name': 'ibis2.q'}),
        "structures_group_pk": Int(5)
}


# In[ ]:


optimize_WorkChain = WorkflowFactory('aiida_cryspy.optimize_structures')
result,node = run.get_node(optimize_WorkChain, **inputs)

