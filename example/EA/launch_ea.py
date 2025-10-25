#!/usr/bin/env python
# coding: utf-8

from aiida.engine import run,submit
from aiida.orm import load_code, Int, Dict, Str
from aiida.plugins import WorkflowFactory
from aiida import load_profile

# AiiDAのプロファイルをロード
load_profile()

# 実行したいマスターWorkChainをAiiDAのプラグインシステムからロード
# (aiida-cryspyプラグインの 'cryspy.ea' という名前で登録されていると仮定)
try:
    EAWorkChain = WorkflowFactory("aiida_cryspy.ea")
except Exception:
    print("Error: Could not load WorkChain 'cryspy.ea'.")
    print("Please ensure your aiida-cryspy plugin is installed and the entry point is correct.")
    exit()

# --- 1. WorkChainへの入力情報を準備 ---

# a) 計算コードをデータベースからロード
try:
    code = load_code("aiida-ase@script_ssh2")
except Exception:
    print("Error: Could not load code 'aiida-ase@script_ssh2'.")
    print("Please make sure the code is set up in AiiDA with 'verdi code setup'.")
    exit()

# b) 共通の計算パラメータをDictノードとして準備
prepend_commands = """
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=2
"""

parameters = Dict(dict={
    'calculator': {
        'args': {
            "load_path": "/home/morii22/.local/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth",
            "device": "cpu",
        },
    },
    "optimizer": {
        "name": "BFGS",
        "run_args": {"fmax": 0.01, "steps": 2000},
        "setup": {"FixSymmetry": True, "FrechetCellFilter": True},
    },
    "extra_imports": [["mattersim.forcefield", "MatterSimCalculator"], "torch"],
    "pre_lines": ["custom_calculator = MatterSimCalculator"],
    "post_lines": [
        "for key, value in results.items():",
        "    if not isinstance(value, numpy.ndarray) and hasattr(value, 'item'):",
        "        results[key] = value.item()",
        "atoms.calc = None"
    ],
    "atoms_getters": ["potential_energy", "forces"],
})

# c) 計算のオプションをDictノードとして準備
options = Dict(dict={
    "resources": {"tot_num_mpiprocs": 2, "parallel_env": "smp"},
    "max_wallclock_seconds": 600,
    "prepend_text": prepend_commands,
    "queue_name": "ibis2.q"
})

# --- 2. WorkChainのビルダーに入力を設定 ---

# WorkChainの入力を作成するための「ビルダー」を取得
builder = EAWorkChain.get_builder()

# defineメソッドで定義した各入力に、準備したAiiDAノードや値を設定
builder.max_generations = Int(2)
builder.cryspy_in_filename = Str("cryspy_in")
builder.code = code
builder.parameters = parameters
builder.options = options


# 実際の計算はAiiDAデーモンがバックグラウンドで管理
# process_node = submit(builder)
process_node = run(builder)

print(f"Successfully submitted {EAWorkChain.__name__}<{process_node.pk}> to the AiiDA daemon.")
print("You can check the status with: verdi process list")
