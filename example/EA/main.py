#!/usr/bin/env python
# coding: utf-8

from aiida import load_profile
# AiiDAプロファイルのロード
load_profile()

from aiida.orm import Dict, Int, Str, load_code
from aiida.plugins import WorkflowFactory
from aiida.engine import submit

ea_WorkChain = WorkflowFactory('aiida_cryspy.ea')

# 実行したい最大世代数
max_generations = 10

# cryspyの入力ファイル名
cryspy_in_filename = "cryspy_in"

# 使用する計算コードのラベル
code_label = "aiida-ase@script_ssh2"

# 計算オプション (Queueやリソースの設定)
options = {
    "resources": {'tot_num_mpiprocs': 2, 'parallel_env': 'smp'},
    "max_wallclock_seconds": 1800,
    "prepend_text": """
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=2
""",
    "queue_name": 'ibis3.q'
}

# MatterSim / Optimizer パラメータ設定
calc_parameters = {
    'calculator': {
        'args': {
            'load_path': '/home/morii22/.local/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth',
            'device': "cpu",
        },
    },
    'optimizer': {
        'name': 'BFGS',
        'run_args': {
            'fmax': 0.01,
            'steps': 3000
        },
        'args': {
            'maxstep': 0.01,
        },
        'setup': {
            'FixSymmetry': True,
            'FrechetCellFilter': True, 
            'ExpCellFilter': False,
            'scalar_pressure': 3, # 目標圧力 (GPa)
        },
    },
    'extra_imports': [
        ['mattersim.forcefield', 'MatterSimCalculator'],
        'torch',
    ],
    'pre_lines': [
        "custom_calculator = MatterSimCalculator",
    ],
    'post_lines': [
        'for key, value in results.items():',
        '    if not isinstance(value, numpy.ndarray) and hasattr(value, "item"):',
        '        results[key] = value.item()',
        'atoms.calc = None'
    ],
    'atoms_getters': [
        'potential_energy',
        'forces',
    ],
}


def main():
    print("Preparing to submit EA_WorkChain to AiiDA daemon...")
    print(f"Max Generations: {max_generations}")
    print(f"Target Pressure: {calc_parameters['optimizer']['setup']['scalar_pressure']} GPa")


    # --- 計算コードとパラメータのAiiDAのノード化 ---
    code = load_code(code_label)
    parameters_node = Dict(dict=calc_parameters)
    options_node = Dict(dict=options)

    # --- EA_WorkChain の入力データ ---
    inputs = {
        "max_generations": Int(max_generations),
        "cryspy_in_filename": Str(cryspy_in_filename),
        "code": code,
        "parameters": parameters_node,
        "options": options_node,
    }

    # --- WorkChainのサブミット ---
    node = submit(ea_WorkChain, **inputs)

    print("\n==================================================")
    print(f"Successfully submitted EA_WorkChain")
    print(f"ea_WorkChain PK: {node.pk}")
    print("==================================================")
    print("The evolutionary algorithm is now running in the background.")
    print("You can monitor the progress by running the following command in your terminal:")
    print("  verdi process list")
    print("To see detailed logs of this run, use:")
    print(f"  verdi process report {node.pk}")


if __name__ == "__main__":
    main()
