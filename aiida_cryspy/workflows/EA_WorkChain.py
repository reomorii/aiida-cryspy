from aiida.engine import WorkChain, ToContext, while_,if_
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.orm import Int, Str, Code, Dict, Group, load_group

# 各WorkChainをFactoryからロード
InitializeWorkChain = WorkflowFactory("aiida_cryspy.initial_structures")
MultiStructureOptimizeWorkChain = WorkflowFactory("aiida_cryspy.optimize_structures")
NextSgWorkChain = WorkflowFactory("aiida_cryspy.next_sg")
PandasFrameData = DataFactory("aiida_cryspy.dataframe")

class EA_WorkChain(WorkChain):
    """
    進化アルゴリズムの全プロセスを管理するマスターWorkChain。
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # --- 入力 ---
        spec.input("max_generations", valid_type=Int, default=lambda: Int(50))
        spec.input("cryspy_in_filename", valid_type=Str, default=lambda: Str("cryspy_in"))
        spec.input("code", valid_type=Code)
        spec.input("parameters", valid_type=Dict)
        spec.input("options", valid_type=Dict)

        # --- 出力 ---
        spec.output("final_structures_group", valid_type=Group, help="最終的に最適化された全構造を含むGroup")
        spec.output("final_results", valid_type=PandasFrameData)

        # --- ワークフロー ---
        spec.outline(
            cls.initialize,
            while_(cls.should_continue_ea)(
                cls.run_optimization,
                cls.run_next_generation,
            ),
            cls.run_final_optimization,
            cls.finalize,
        )

    def initialize(self):
        """
        初期化ステップ。Groupの作成と初期構造の生成を行う。
        """
        # 1. この実行専用のGroupを作成し、そのpkをコンテキストに保存
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        group_label = f"cryspy-ea-run/{timestamp}"
        group = Group(label=group_label).store()
        self.ctx.group_pk = group.pk
        self.report(f"Created Group<{group.pk}> with label '{group.label}' for this run.")

        # 2. 初期化WorkChainを呼び出す
        inputs = {'cryspy_in_filename': self.inputs.cryspy_in_filename}
        running = self.submit(InitializeWorkChain, **inputs)
        self.report(f"Launched InitializeWorkChain<{running.pk}>")
        return ToContext(initialization=running)

    def should_continue_ea(self):
        """
        EAのループを継続するかどうかを判定する。
        """

        # ループの初回は 'initialization' から世代データを取得する
        if "next_generation" not in self.ctx:
            current_gen = self.ctx.initialization.outputs.detail_data.ea_data[0]
        # 2回目以降は 'next_generation' の出力から取得する
        else:
            current_gen = self.ctx.next_generation.outputs.detail_data.ea_data[0]

        max_gen = self.inputs.max_generations.value
        self.report(f"Current generation: {current_gen}, Max generations: {max_gen}")
        return current_gen < max_gen

    def run_optimization(self):
        """
        構造最適化WorkChainを呼び出す。
        """
        # 前のステップの出力を取得
        if "initialization" in self.ctx:  # 初回ループ
            init_outputs = self.ctx.initialization.outputs
        else:  # 2回目以降のループ
            init_outputs = self.ctx.next_generation.outputs

        inputs = {
            "initial_structures": init_outputs.initial_structures,
            "id_queueing": init_outputs.id_queueing,
            "rslt_data": init_outputs.rslt_data,
            "cryspy_in": init_outputs.cryspy_in,
            "detail_data": init_outputs.detail_data,
            "code": self.inputs.code,
            "parameters": self.inputs.parameters,
            "options": self.inputs.options,
            # ★ GroupのPKを渡す
            "structures_group_pk": Int(self.ctx.group_pk),
        }

        running = self.submit(MultiStructureOptimizeWorkChain, **inputs)
        self.report(f"Launched MultiStructureOptimizeWorkChain<{running.pk}>")
        return ToContext(optimization=running)

    def run_next_generation(self):
        """
        次世代構造生成WorkChainを呼び出す。
        """
        opt_outputs = self.ctx.optimization.outputs

        # 前のステップの初期構造データを取得
        if 'initialization' in self.ctx:
            initial_structures_node = self.ctx.initialization.outputs.initial_structures
            detail_data_node = self.ctx.initialization.outputs.detail_data
            cryspy_in_node = self.ctx.initialization.outputs.cryspy_in
        else:
            initial_structures_node = self.ctx.next_generation.outputs.next_structures
            detail_data_node = self.ctx.next_generation.outputs.detail_data
            cryspy_in_node = self.ctx.next_generation.outputs.cryspy_in


        inputs = {
            "initial_structures": initial_structures_node,
            "rslt_data": opt_outputs.rslt_data,
            "detail_data": detail_data_node,
            "cryspy_in": cryspy_in_node,
            # ★ GroupのPKを渡す
            "structures_group_pk": Int(self.ctx.group_pk),
        }

        running = self.submit(NextSgWorkChain, **inputs)
        self.report(f"Launched NextSgWorkChain<{running.pk}>")
        return ToContext(next_generation=running)

    def run_final_optimization(self):
        """最終世代の最適化のみを実行する。"""
        self.report(f"Running final optimization for generation {self.inputs.max_generations.value}.")

        # 最後のrun_next_generation (49世代目) の結果を使う
        last_outputs = self.ctx.next_generation.outputs

        inputs = {
            'initial_structures': last_outputs.next_structures,
            'id_queueing': last_outputs.id_queueing,
            'rslt_data': last_outputs.rslt_data,
            'cryspy_in': last_outputs.cryspy_in,
            'detail_data': last_outputs.detail_data,
            "code": self.inputs.code,
            "parameters": self.inputs.parameters,
            "options": self.inputs.options,
            "structures_group_pk": Int(self.ctx.group_pk),
        }
        running = self.submit(MultiStructureOptimizeWorkChain, **inputs)
        return ToContext(final_optimization=running)

    def finalize(self):
        """
        最終結果をWorkChainの出力に設定する。
        """
        self.report("Evolutionary algorithm finished.")
        group = load_group(pk=self.ctx.group_pk)
        final_results = self.ctx.final_optimization.outputs.rslt_data

        self.out('final_structures_group', group)
        self.out('final_results', final_results)
