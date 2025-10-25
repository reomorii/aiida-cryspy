#!/usr/bin/env python
# coding: utf-8

import datetime
from aiida.orm import Dict, Int, Str, Group, load_code
from aiida.engine import run
from aiida.plugins import WorkflowFactory

# --- AiiDAプロファイルのロード ---
from aiida import load_profile
load_profile()

# ==============================================================================
# ▼▼▼ ユーザー設定項目 ▼▼▼
# ==============================================================================
# 実行したい世代数
num_generations = 10

# cryspyの入力ファイル名
cryspy_in_filename = Str("cryspy_in")

# 使用する計算コードのラベル
code_label = "aiida-ase@script_ssh2"

# 計算オプション
options = {
    "resources": {'tot_num_mpiprocs': 4, 'parallel_env': 'smp'},
    "max_wallclock_seconds": 600,
    "prepend_text": """
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=2
""",
    "queue_name": 'ibis2.q'
}
# ==============================================================================
# ▲▲▲ ユーザー設定項目 ▲▲▲
# ==============================================================================


# --- WorkChainの読み込み ---
initialize_WorkChain = WorkflowFactory('aiida_cryspy.initial_structures')
optimize_WorkChain = WorkflowFactory('aiida_cryspy.optimize_structures')
next_sg_WorkChain = WorkflowFactory('aiida_cryspy.next_sg')


# --- グループの動的作成 ---
# 実行ごとにユニークなグループを作成し、構造を保存する
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
group_label = f"cryspy-ea-run/{timestamp}"
group = Group(label=group_label).store()
structures_group_pk = Int(group.pk)
print(f"✅ Created Group <PK: {group.pk}> with label '{group_label}' for this run.")


# --- 計算パラメータの設定 ---
code = load_code(code_label)
parameters = Dict(dict={
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
            'steps': 2000
        },
        'setup': {
            'FixSymmetry': True,
            'FrechetCellFilter': True,
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
})

# --- Step 1: 初期構造の生成 (ループ外で一度だけ実行) ---
print("🚀 Starting Step 1: Initial structure generation...")
init_result, init_node = run.get_node(initialize_WorkChain, cryspy_in_filename=cryspy_in_filename)
print(f"Initial structure generation finished. WorkChain <PK: {init_node.pk}>")

# 次の計算に渡すノードを準備
initial_structures_node = init_result['initial_structures']
rslt_data_node = init_result['rslt_data']
cryspy_in_node = init_result['cryspy_in']
detail_data_node = init_result['detail_data']
id_queueing_node = init_result['id_queueing']

# --- Step 2: 進化的アルゴリズムのループ ---
for gen in range(1, num_generations + 1):
    print(f"\n世代 {gen}/{num_generations} を開始します。")
    print("--------------------------------------------------")

    # --- 構造最適化 ---
    print(f"🧬 [Generation {gen}] Running structure optimization...")
    opt_inputs = {
        'initial_structures': initial_structures_node,
        'rslt_data': rslt_data_node,
        'cryspy_in': cryspy_in_node,
        'detail_data': detail_data_node,
        'id_queueing': id_queueing_node,
        "code": code,
        "parameters": parameters,
        "options": Dict(dict=options),
        "structures_group_pk": structures_group_pk
    }
    opt_result, opt_node = run.get_node(optimize_WorkChain, **opt_inputs)
    print(f"Optimization finished. WorkChain <PK: {opt_node.pk}>")

    # 最終世代でなければ、次の世代を生成
    if gen < num_generations:
        # --- 次世代構造の生成 ---
        print(f" [Generation {gen}] Generating next structures...")
        next_sg_inputs = {
            'initial_structures': initial_structures_node, # 最適化の"入力"となった構造
            'rslt_data': opt_result['rslt_data'],           # 最適化の"結果"
            'detail_data': detail_data_node,
            'cryspy_in': cryspy_in_node,
            'structures_group_pk': structures_group_pk,
        }
        next_sg_result, next_sg_node = run.get_node(next_sg_WorkChain, **next_sg_inputs)
        print(f"Next structure generation finished. WorkChain <PK: {next_sg_node.pk}>")

        # --- 次のループのためにノードを更新 ---
        initial_structures_node = next_sg_result['next_structures']
        rslt_data_node = next_sg_result['rslt_data']
        detail_data_node = next_sg_result['detail_data']
        id_queueing_node = next_sg_result['id_queueing']
    else:
        print("\n 全ての世代の計算が完了しました。")