# aiida-cryspy

AiiDA上でのCrySPYのワークフローを自動化・管理するためのパッケージです。



## 前提条件

本パッケージを利用するには、AiiDA (aiida-core) をセットアップし、プロファイルを作成する必要があります． AiiDA自体のセットアップについては、AiiDA公式ドキュメント ( https://aiida.readthedocs.io/ ) を参照してください。



## インストール方法

リポジトリをクローンし、開発モード（または通常のpip）でインストールします。

```bash
git clone https://github.com/reomorii/aiida-cryspy.git
```

```bash
cd aiida-cryspy
pip install -e .
```
インストール時に、以下の依存パッケージも自動的にインストールされます。

- aiida-core
- aiida-ase (GitHubリポジトリより)
- csp-cryspy (GitHubリポジトリより)

※ なお、`aiida-ase` および `csp-cryspy` は、本パッケージ（`aiida-cryspy`）で正常に動作させるため、元のリポジトリをフォークして独自の改良を加えたバージョンを使用しています。

インストールが成功すると、AiiDAが自動的にWorkChainやData型を認識します。確認するには以下のコマンドを実行してください：

```bash
verdi plugin list aiida.workflows
verdi plugin list aiida.data
```



## 実行方法

exampleフォルダ配下にあるサンプルを実行する手順です。

### 注意：作業ディレクトリについて
実行ファイル（main.pyなど）を動かす際、カレントディレクトリが出力ファイルの参照先（作業ディレクトリ）になります。そのため、必ず実行するディレクトリでデーモン（Daemon）を起動してください。



### ステップ1: デーモンの起動
example などの、計算を実行したいディレクトリへ移動し、デーモンを起動（または再起動）します。

```bash
cd path/to/your/example_directory
verdi daemon start
```



### ステップ2: スクリプトの実行
```bash
python3 main.py
```


## プラグイン


### WorkChains (aiida.workflows)

| プラグイン名 | 呼び出しパス | 概要 |
| :--- | :--- | :--- |
| 初期構造生成 | aiida_cryspy.initial_structures | 初期構造を作成 |
| 構造最適化 | aiida_cryspy.optimize_structures | 構造を最適化 |
| 次世代生成 | aiida_cryspy.next_sg | 次の世代の構造を生成 |
| 進化的アルゴリズム | aiida_cryspy.ea | EA（進化的アルゴリズム）を実行 |

### Data Types (aiida.data)

- aiida_cryspy.dataframe (Pandas DataFrameの保存用)
- aiida_cryspy.ea_data （EAについてのデータの保存用）
- aiida_cryspy.rin_data (cryspy.inについてのデータの保存用)
- aiida_cryspy.structurecollection (構造データの保存用)

