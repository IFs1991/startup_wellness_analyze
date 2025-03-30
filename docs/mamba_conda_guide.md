# mamba/conda環境活用ガイド

## 概要

このガイドでは、スタートアップ分析プラットフォームにおける mamba/conda/pip 複合環境の使い方と
ベストプラクティスについて説明します。この環境は、パッケージの依存関係解決の高速化と
システム全体のパフォーマンス最適化を目的としています。

## mamba/condaとは

### conda
- Python および他の言語のパッケージマネージャー
- 依存関係の解決と環境の分離を提供
- バイナリパッケージの配布により、コンパイルの手間を省く

### mamba
- condaの高速な代替実装
- C++で書かれており、依存関係の解決が大幅に高速化
- conda互換のAPIを提供しつつ、パフォーマンスを向上

### pip との連携
- mambaとpipを併用することで、最大のパッケージカバレッジを実現
- conda-forgeで提供されていないパッケージをpipで補完
- 環境内でのpip管理により、依存関係の衝突を防止

## セットアップ手順

### 1. 初期セットアップ

プロジェクトのルートディレクトリにある `setup_environment.sh` スクリプト（Linuxまたは macOS）または
`setup_environment.ps1` スクリプト（Windows）を実行して、環境を構築します。

```bash
# Linuxまたは macOS
chmod +x setup_environment.sh
./setup_environment.sh

# Windows (PowerShell)
# PowerShellでスクリプトを実行するには、セキュリティポリシーの変更が必要な場合があります
.\setup_environment.ps1
```

このスクリプトは以下の処理を行います：
- miniforgeのインストール確認（なければインストール）
- mambaのインストール
- 環境定義ファイル（environment.yml）から環境の作成
- 東京リージョンの割引時間帯設定の確認
- GCPプロジェクトIDの設定（オプション）
- 環境診断の実行（オプション）

### 2. 環境の有効化

```bash
conda activate causal-analytics
```

### 3. 環境の更新

依存関係が変更された場合は、以下のコマンドで環境を更新します：

```bash
mamba env update -f environment.yml
```

## 日常的な使用方法

### パッケージの追加

環境に新しいパッケージを追加する場合は、以下のコマンドを使用します：

```bash
# conda-forgeから追加（優先）
mamba install -c conda-forge パッケージ名

# pipを使って追加（conda-forgeにない場合）
pip install パッケージ名
```

ただし、**重要**: 可能な限り、直接 `mamba install` や `pip install` を使うのではなく、
`environment.yml` ファイルを更新してから `mamba env update` を実行することをお勧めします。
これにより、チーム全体で一貫した環境を維持できます。

#### conda vs pip の選択基準

1. **基本原則**: まず conda-forge でパッケージを探し、見つからない場合またはバージョン要件を満たさない場合のみ pip を使用
2. **環境.yml での管理**:
   ```yaml
   dependencies:
     # conda-forge パッケージ
     - numpy>=2.0.0
     # pip パッケージ
     - pip:
       - tensorflow>=2.8.0
   ```
3. **優先順位**:
   - バイナリ依存関係を持つパッケージ（TensorFlow, PyTorch など）: conda を優先
   - 純粋な Python パッケージ: どちらでも良いが、conda-forge で入手可能なら conda を推奨
   - 最新バージョンが必要な場合: pip が最新の場合が多い

### パッケージのアップデート

```bash
# 単一パッケージのアップデート
mamba update パッケージ名

# 環境内のすべてのパッケージをアップデート
mamba update --all

# pip経由でインストールしたパッケージのアップデート
pip install --upgrade パッケージ名
```

### 環境情報の確認

```bash
# インストールされているパッケージ一覧
mamba list

# pip経由でインストールされたパッケージの一覧
pip list

# 環境の情報をエクスポート
mamba env export > environment_snapshot.yml

# 明示的な依存関係のみエクスポート
mamba env export --from-history > environment_minimal.yml
```

## 連合学習最適化とGCPコスト削減

### GCP東京リージョンの割引時間帯

当プロジェクトでは、東京リージョン(`asia-northeast1`)の割引時間帯を活用するために、
連合学習ワークフローを最適化しています。

#### 割引時間帯（日本時間）
- **平日**: 22:00-08:00 (JST)
- **週末**: 終日

これらの時間帯に計算負荷の高いタスクをスケジュールすることで、**約20%のコスト削減**が可能です。

#### 連合学習の最適スケジューリング

`backend/federated_learning/scheduler/optimal_scheduling.py` モジュールを使用して、
連合学習タスクを割引時間帯に自動的にスケジュールできます：

```python
from backend.federated_learning.scheduler.optimal_scheduling import FederatedLearningScheduler

# スケジューラの初期化
scheduler = FederatedLearningScheduler(
    project_id="your-project-id",
    location="asia-northeast1"
)

# 連合学習ジョブをスケジュール（自動的に割引時間帯に最適化）
scheduler.schedule_federated_learning(
    job_name="weekly_model_training",
    endpoint_url="https://your-service/api/federated/train",
    body={"model_type": "collaborative", "rounds": 10},
    estimated_duration_minutes=180  # 予想される実行時間（分）
)
```

#### GCPコスト分析と最適化

`scripts/optimize_gcp_costs.py` スクリプトを実行して、現在のGCP環境のコスト分析とリソース最適化の提案を取得できます：

```bash
# GCPプロジェクトIDを指定して実行
python scripts/optimize_gcp_costs.py --project-id your-project-id

# 出力レポートの保存先を指定
python scripts/optimize_gcp_costs.py --project-id your-project-id --output cost_report.json
```

このスクリプトは以下の最適化を提案します：
- 低使用率インスタンスのリサイズまたは停止スケジュール
- スポットインスタンスへの変換機会
- 連合学習ジョブの割引時間帯へのスケジュール最適化

## 大規模データ処理

### Daskを活用した分散処理

`backend/utils/dask_optimizer.py` モジュールを使用して、大規模データの処理を最適化できます：

```python
from backend.utils.dask_optimizer import DaskOptimizer

# オプティマイザのインスタンス化（リソース使用量を自動調整）
optimizer = DaskOptimizer()

# コンテキストマネージャとして使用
with DaskOptimizer() as optimizer:
    # 大規模データフレームの処理
    result_df = optimizer.process_large_dataset(large_df, processing_function)

    # ファイル読み込みの最適化
    data = optimizer.read_csv_optimized("large_file.csv", memory_efficient=True)

    # データ型の最適化によるメモリ使用量削減
    optimized_df = optimizer.optimize_dataframe_dtypes(df)
```

### メモリ最適化テクニック

大規模データセット（10,000件/月の120次元データ）を処理する際のメモリ最適化テクニック：

1. **データ型の最適化**

    ```python
    # 手動による最適化
    df["int_column"] = df["int_column"].astype(np.int32)
    df["float_column"] = df["float_column"].astype(np.float32)
    df["category_column"] = df["category_column"].astype("category")

    # または DaskOptimizer を使用
    from backend.utils.dask_optimizer import DaskOptimizer
    df_optimized = DaskOptimizer.optimize_dataframe_dtypes(df)
    ```

2. **チャンク処理**

    ```python
    # 大きなCSVファイルを分割して読み込む
    chunk_size = 100000  # 一度に処理する行数
    chunks = []

    for chunk in pd.read_csv("large_file.csv", chunksize=chunk_size):
        # 各チャンクを処理
        processed_chunk = process_function(chunk)
        chunks.append(processed_chunk)

    # 結果を結合
    result = pd.concat(chunks)
    ```

3. **不要なカラムの削除**

    ```python
    # 必要なカラムのみ選択
    df = df[["important_col1", "important_col2", "important_col3"]]

    # または不要なカラムを削除
    df.drop(columns=["unused_col1", "unused_col2"], inplace=True)
    ```

## ベストプラクティス

### 1. 環境管理

- プロジェクト全体で一貫したバージョンを使用するために、`environment.yml` を定期的に更新する
- 互換性の問題を避けるため、明示的なバージョン制約を指定する
- `mamba list --explicit > explicit-spec.txt` で完全な仕様をエクスポートし、バージョン管理する

### 2. パフォーマンス最適化

- `dask` を活用して大規模データ処理を並列化する
- pandas DataFrameのデータ型を最適化してメモリ使用量を削減する
- 処理パイプラインを構築して、中間結果をキャッシュする

### 3. 連合学習のコスト最適化

- 東京リージョンの割引時間帯に合わせてスケジュール設定を行う
- 定期的なコスト分析を実行して、リソース使用状況を監視する
- 必要に応じて、インスタンスサイズの調整やスポットインスタンスの活用を検討する

### 4. pip と conda の併用ベストプラクティス

- 基本環境は conda で構築し、専門的なパッケージを pip で追加する
- pipパッケージは必ず environment.yml の pip セクションで管理する
- pip インストールは常に conda 環境がアクティブな状態で行う
- 環境作成時には conda パッケージを先にインストールし、その後 pip パッケージをインストールする

## スケーリングガイドライン

データ量の増加に応じた段階的な環境拡張計画：

### 〜500件/月（現在）
- 基本的な環境設定と最適化
- 東京リージョンの割引時間帯の活用開始
- 連合学習のスケジュール最適化

### 500〜2,000件/月
- Daskを使用した並列処理の導入
- 次元削減前処理の実装
- スポットインスタンスの活用

### 2,000〜5,000件/月
- 分散処理アーキテクチャの導入
- インクリメンタル処理パイプラインの実装
- データストレージの階層化

### 5,000〜10,000件/月
- フル分散処理環境
- 高度なキャッシング戦略
- 複雑なオートスケーリング

## トラブルシューティング

### 依存関係の競合

依存関係の競合が発生した場合は、以下の手順を試してください：

1. 依存関係の明示的な表示:
   ```bash
   mamba list --explicit
   ```

2. 環境の再構築:
   ```bash
   conda deactivate
   conda env remove -n causal-analytics
   mamba env create -f environment.yml
   ```

3. チャンネル優先度の調整:
   ```yaml
   # environment.yml
   channels:
     - conda-forge  # 優先度高
     - defaults     # 優先度中
     - その他        # 優先度低
   ```

### メモリ使用量の問題

メモリ消費が多い場合は、以下の対策を検討してください：

1. Daskを使用した処理の並列化
2. pandas DataFrameのデータ型最適化
3. 大きなファイルの分割処理

```python
# DataFrameのメモリ使用量を最適化
from backend.utils.dask_optimizer import DaskOptimizer
df_optimized = DaskOptimizer.optimize_dataframe_dtypes(df)

# メモリ使用量の確認
def check_memory_usage(df):
    usage = df.memory_usage(deep=True).sum()
    print(f"メモリ使用量: {usage / (1024**2):.2f} MB")

check_memory_usage(df)
check_memory_usage(df_optimized)
```

### pip と conda の依存関係衝突

pip と conda の依存関係で衝突が発生した場合：

1. conda パッケージを先にインストールし、pip パッケージを後でインストール
   ```bash
   mamba install -c conda-forge pandas numpy
   pip install special-package
   ```

2. 衝突するバージョンを特定して明示的に指定
   ```yaml
   dependencies:
     - numpy=1.20.3  # 特定バージョンを指定
     - pip:
       - package-requiring-numpy==1.20.3
   ```

3. 完全に互換性のない場合は、別の環境を作成して分離する

## GCP割引時間帯の最大活用方法

### 自動スケジューリングの設定

1. **Cloud Schedulerを使用したタスク実行**

   ```python
   # コスト最適化スクリプトのスケジュール実行
   from google.cloud import scheduler_v1

   client = scheduler_v1.CloudSchedulerClient()
   parent = f"projects/{project_id}/locations/{location}"

   job = {
       "name": f"{parent}/jobs/cost_optimization_job",
       "schedule": "0 0 * * 1",  # 毎週月曜日の午前0時
       "time_zone": "Asia/Tokyo",
       "http_target": {
           "uri": "https://your-service/api/optimize",
           "http_method": "POST"
       }
   }

   client.create_job(parent=parent, job=job)
   ```

2. **最適な時間帯にバッチジョブを実行**

   ```bash
   # 平日の場合: 22:00以降にスケジュール
   # 週末の場合: 任意の時間にスケジュール
   python -c "
   import datetime
   now = datetime.datetime.now()
   is_weekend = now.weekday() >= 5
   is_discount_hour = now.hour >= 22 or now.hour < 8

   if is_weekend or is_discount_hour:
       print('割引時間帯です。ジョブを実行します。')
       # ここにジョブ実行コードを記述
   else:
       print('割引時間帯ではありません。ジョブをスケジュールします。')
       # ここにスケジューリングコードを記述
   "
   ```

## 参考リソース

- [Miniforge GitHub](https://github.com/conda-forge/miniforge)
- [Mamba Documentation](https://mamba.readthedocs.io/)
- [Conda-forge](https://conda-forge.org/)
- [Dask Documentation](https://docs.dask.org/)
- [GCP 割引](https://cloud.google.com/compute/docs/regions-zones/)
- [TensorFlow Federated](https://www.tensorflow.org/federated)
- [GCP Cost Optimization Best Practices](https://cloud.google.com/architecture/best-practices-for-running-cost-effective-kubernetes)