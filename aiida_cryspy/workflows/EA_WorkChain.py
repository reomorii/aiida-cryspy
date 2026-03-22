from aiida.engine import WorkChain, ToContext, while_
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.orm import Int, Str, Code, Dict, load_group

# 各WorkChainをインポート
InitializeWorkChain = WorkflowFactory("aiida_cryspy.initial_structures")
MultiStructureOptimizeWorkChain = WorkflowFactory("aiida_cryspy.optimize_structures")
NextSgWorkChain = WorkflowFactory("aiida_cryspy.next_sg")
PandasFrameData = DataFactory("aiida_cryspy.dataframe")

class EA_WorkChain(WorkChain):
    """
    AiiDA-CrySPYの進化的アルゴリズム(EA)全体を統括するWorkChain。
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # --- Inputs ---
        spec.input("max_generations", valid_type=Int, default=lambda: Int(50))
        spec.input("cryspy_in_filename", valid_type=Str, default=lambda: Str("cryspy_in"))
        spec.input("code", valid_type=Code)
        spec.input("parameters", valid_type=Dict)
        spec.input("options", valid_type=Dict)

        # --- Outputs ---
        spec.output("optimized_structures_group_pk", valid_type=Int, help="全世代の最適化済み構造が蓄積されたGroupのPK")
        spec.output("final_rslt_data", valid_type=PandasFrameData, help="最終結果データ")

        # --- Outline ---
        spec.outline(
            cls.run_initialize,         # 1. 初期化実行
            cls.setup_initial_context,  # 2. 初期化結果をコンテキストにセット
            while_(cls.should_continue_ea)(
                cls.run_optimization,   # 3. 最適化実行
                cls.update_opt_data,    # 4. 最適化結果をコンテキストに反映
                cls.run_next_generation,# 5. 次世代生成実行
                cls.update_next_data,   # 6. 次世代生成結果をコンテキストに反映
            ),
            cls.run_final_optimization, # 7. 最終世代の最適化
            cls.finalize,               # 8. 完了処理
        )

    def run_initialize(self):
        """初期構造生成 WorkChainの実行"""
        self.report("[Step 1] Running InitializeWorkChain...")
        inputs = {'cryspy_in_filename': self.inputs.cryspy_in_filename}
        running = self.submit(InitializeWorkChain, **inputs)
        return ToContext(init_wc=running)

    def setup_initial_context(self):
        """InitializeWorkChainが作成したGroupのPKやデータをコンテキストに保存"""
        outputs = self.ctx.init_wc.outputs
        
        # 世代ごとに更新される変数
        self.ctx.current_structures_group_pk = outputs.initial_structures_group_pk
        self.ctx.rslt_data = outputs.rslt_data
        self.ctx.detail_data = outputs.detail_data
        self.ctx.id_queueing = outputs.id_queueing
        
        # 全世代で不変の変数（常に使い回す）
        self.ctx.optimized_structures_group_pk = outputs.optimized_structures_group_pk
        self.ctx.cryspy_in = outputs.cryspy_in

    def should_continue_ea(self):
        """世代数の判定"""
        current_gen = self.ctx.detail_data.ea_data[0]
        max_gen = self.inputs.max_generations.value
        self.report(f"--- Generation {current_gen} / {max_gen} ---")
        return current_gen < max_gen

    def run_optimization(self):
        """構造最適化 WorkChainの実行"""
        inputs = {
            "initial_structures_group_pk": self.ctx.current_structures_group_pk,
            "optimized_structures_group_pk": self.ctx.optimized_structures_group_pk, # 蓄積用Groupを渡す
            "rslt_data": self.ctx.rslt_data,
            "cryspy_in": self.ctx.cryspy_in,
            "detail_data": self.ctx.detail_data,
            "id_queueing": self.ctx.id_queueing,
            "code": self.inputs.code,
            "parameters": self.inputs.parameters,
            "options": self.inputs.options,
        }
        running = self.submit(MultiStructureOptimizeWorkChain, **inputs)
        return ToContext(opt_wc=running)

    def update_opt_data(self):
        """最適化後の結果でコンテキストの rslt_data を更新"""
        self.ctx.rslt_data = self.ctx.opt_wc.outputs.rslt_data

    def run_next_generation(self):
        """次世代構造生成 WorkChainの実行"""
        inputs = {
            "initial_structures_group_pk": self.ctx.current_structures_group_pk, # 親(最適化前)のGroup
            "optimized_structures_group_pk": self.ctx.optimized_structures_group_pk, # 親(最適化後)のGroup
            "rslt_data": self.ctx.rslt_data, # 更新された最新の結果
            "detail_data": self.ctx.detail_data,
            "cryspy_in": self.ctx.cryspy_in,
        }
        running = self.submit(NextSgWorkChain, **inputs)
        return ToContext(next_wc=running)

    def update_next_data(self):
        """NextSgWorkChainが新しく作成したGroupのPKやデータでコンテキストを上書き（次のループの準備）"""
        outputs = self.ctx.next_wc.outputs
        self.ctx.current_structures_group_pk = outputs.next_structures_group_pk
        self.ctx.rslt_data = outputs.rslt_data
        self.ctx.detail_data = outputs.detail_data
        self.ctx.id_queueing = outputs.id_queueing

    def run_final_optimization(self):
        """ループを抜けた後、最終世代の最適化のみを実行"""
        self.report("Running final optimization...")
        inputs = {
            "initial_structures_group_pk": self.ctx.current_structures_group_pk,
            "optimized_structures_group_pk": self.ctx.optimized_structures_group_pk,
            "rslt_data": self.ctx.rslt_data,
            "cryspy_in": self.ctx.cryspy_in,
            "detail_data": self.ctx.detail_data,
            "id_queueing": self.ctx.id_queueing,
            "code": self.inputs.code,
            "parameters": self.inputs.parameters,
            "options": self.inputs.options,
        }
        running = self.submit(MultiStructureOptimizeWorkChain, **inputs)
        return ToContext(final_opt_wc=running)

    def finalize(self):
        """最終結果の出力"""
        self.report("Evolutionary algorithm finished completely.")
        self.out('optimized_structures_group_pk', self.ctx.optimized_structures_group_pk)
        self.out('final_rslt_data', self.ctx.final_opt_wc.outputs.rslt_data)