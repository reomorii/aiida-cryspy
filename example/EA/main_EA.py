#!/usr/bin/env python
# coding: utf-8

from aiida.orm import Dict, load_code
from aiida.engine import run
from aiida.plugins import WorkflowFactory
from aiida import load_profile

load_profile()

DEFAULT_MAX_GENERATIONS = 50

def main():
    # --- 1. 初期化ステップ (ループの外で一度だけ実行) ---
    print("--- start aiida-cryspy ---")
    initialize_WorkChain = WorkflowFactory('aiida_cryspy.initial_structures')
    # run関数は (result, node) のタプルを直接返す
    result_init = run(initialize_WorkChain, cryspy_in_filename="cryspy_in")

    # ループで使う「状態変数」を初期化する
    # これらの変数は、ループの各世代で新しい値に更新されていく
    initial_structures_node = result_init['initial_structures']
    opt_structures_node = result_init['opt_structures']
    rslt_data_node = result_init['rslt_data']
    id_queueing_node = result_init['id_queueing']
    detail_data_node = result_init['detail_data']
    cryspy_in_node = result_init['cryspy_in'] # これは世代を通して不変

    print("--- Initial structure generation completed ---")

    # --- 共通の入力情報を準備 ---
    code = load_code("aiida-ase@script_ssh2")
    prepend_commands = """
    export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
    export OMP_NUM_THREADS=2
    """

    parameters = Dict(dict={
        'calculator': {
            'args': {
                'load_path': '/home/morii22/.local/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth',
                'device': "cpu",
            },
        },
        'optimizer': {
            'name': 'BFGS',
            'run_args': {'fmax': 0.01, 'steps': 2000},
            'setup': {'FixSymmetry': True, 'FrechetCellFilter': True},
        },
        'extra_imports': [['mattersim.forcefield', 'MatterSimCalculator'], 'torch'],
        'pre_lines': ["custom_calculator = MatterSimCalculator"],
        'post_lines': [
            'for key, value in results.items():',
            '    if not isinstance(value, numpy.ndarray) and hasattr(value, "item"):',
            '        results[key] = value.item()',
            'atoms.calc = None'
        ],
        'atoms_getters': ['potential_energy', 'forces'],
    })
    options = Dict(dict={
        "resources": {'tot_num_mpiprocs': 2, 'parallel_env': 'smp'},
        'max_wallclock_seconds': 600,
        'prepend_text': prepend_commands
    })

    # --- 2. メインループ (世代交代を繰り返す) ---
    for gen in range(1, DEFAULT_MAX_GENERATIONS + 1):
        print(f"\n--- generation {gen}/{DEFAULT_MAX_GENERATIONS} start ---")

        # --- 2a. 構造最適化 ---
        print(f"  Step {gen}-optimization: ")
        optimize_inputs = {
            'initial_structures': initial_structures_node, # 現在の世代の入力
            'opt_structures': opt_structures_node,         # 前の世代の結果
            'id_queueing': id_queueing_node,               # 現在の世代の入力
            'rslt_data': rslt_data_node,                   # 前の世代の結果
            'cryspy_in': cryspy_in_node,
            'detail_data': detail_data_node,               # 現在の世代の入力
            "code": code,
            "parameters": parameters,
            "options": options
        }

        optimize_WorkChain = WorkflowFactory('aiida_cryspy.optimize_structures')
        result_opt = run(optimize_WorkChain, **optimize_inputs)
        print("    Optimization completed")

        opt_structures_node = result_opt['opt_struc_data']  # 最適化された構造を次の世代の初期構造として使用

        # --- 2b. 次世代構造の生成 ---
        print(f"  Step {gen}-next_generation: ")

        next_sg_inputs = {
            'initial_structures': initial_structures_node, # この世代の初期構造
            'opt_structures': opt_structures_node, # ★最適化の「結果」を使う
            'rslt_data': result_opt['rslt_data'],         # ★最適化の「結果」を使う
            'detail_data': detail_data_node,     # ★最適化の「結果」を使う
            'cryspy_in': cryspy_in_node,
        }

        next_sg_WorkChain = WorkflowFactory('aiida_cryspy.next_sg')
        result_next_sg = run(next_sg_WorkChain, **next_sg_inputs)
        print("    Next generation completed.")

        # --- 2c. ★状態変数の更新 (次のループの準備) ---
        # next_sg の結果を使って、次の世代で使われる変数を「上書き」する

        initial_structures_node = result_next_sg['next_structures']
        rslt_data_node = result_next_sg['rslt_data']
        id_queueing_node = result_next_sg['id_queueing']
        detail_data_node = result_next_sg['detail_data']

    print("\n--- Done All Structures ---")

if __name__ == "__main__":
    main()