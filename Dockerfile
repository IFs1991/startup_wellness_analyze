# ====================================
# ビルドステージ
# ====================================
FROM condaforge/mambaforge:latest AS builder

LABEL maintainer="startup-wellness-team"
LABEL description="スタートアップウェルネス分析プラットフォーム - 東京リージョン割引時間帯最適化版"

# 環境変数の設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # 依存関係インストールの並列化
    PIP_DEFAULT_TIMEOUT=100 \
    # メモリ使用量最適化
    MALLOC_TRIM_THRESHOLD_=65536 \
    PYTHONMALLOC=malloc \
    # 割引時間帯設定（東京リージョン）
    DISCOUNT_HOURS_START=22 \
    DISCOUNT_HOURS_END=8 \
    WEEKEND_DISCOUNT=true \
    REGION=asia-northeast1 \
    # daskワーカー設定
    DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.85 \
    DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.90 \
    DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.95 \
    DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.99

# ワークディレクトリの設定
WORKDIR /app

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# environment.ymlをコピーして依存関係をインストール
COPY backend/environment.yml .
RUN mamba env update -n base -f environment.yml && \
    mamba clean -afy && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.pyc' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete

# アプリケーションコードとスクリプトをコピー
COPY ./backend/ /app/backend/
COPY ./scripts/ /app/scripts/
# ルートの entrypoint.sh は不要なので削除

# backend/app/entrypoint.sh に実行権限を付与
RUN chmod +x /app/backend/app/entrypoint.sh

# メモリ使用量監視コマンドの作成
RUN mkdir -p /app/scripts && \
    echo '#!/bin/bash\nps -o pid,user,%mem,command ax | sort -b -k3 -r' > /app/scripts/memory_usage.sh && \
    chmod +x /app/scripts/memory_usage.sh

# entrypoint.shをbackendルートにもコピー
RUN cp /app/backend/app/entrypoint.sh /app/backend/entrypoint.sh && \
    chmod +x /app/backend/entrypoint.sh

# ====================================
# ランタイムイメージ ステージ
# ====================================
FROM condaforge/miniforge3:latest

# 環境変数の設定
ENV TZ=Asia/Tokyo
# PYTHONPATHに /app と /app/backend を追加 (main.py や core などをインポート可能にするため)
ENV PYTHONPATH=/app:/app/backend
ENV DEBIAN_FRONTEND=noninteractive
# Conda環境をデフォルトで有効にする設定
ENV PATH /opt/conda/bin:$PATH

# タイムゾーンの設定とランタイムに必要なシステム依存関係のインストール
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    # 基本システムツール
    ca-certificates \
    # 基本フォント
    fonts-dejavu \
    # デバッグ用ツール
    curl \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ワークディレクトリの設定
WORKDIR /app

# 一般ユーザーの作成 (ランタイムステージで作成)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# ビルドステージから Conda 環境をコピー (所有者を appuser に変更)
COPY --from=builder --chown=appuser:appuser /opt/conda /opt/conda
# ビルドステージのアプリケーションコードをコピー (所有者を appuser に変更)
COPY --from=builder --chown=appuser:appuser /app/backend /app/backend
# ビルドステージのスクリプトをコピー (所有者を appuser に変更)
COPY --from=builder --chown=appuser:appuser /app/scripts /app/scripts

# 永続データや認証情報用のディレクトリを作成 (所有者を appuser に変更)
RUN mkdir -p /app/data /app/credentials && \
    chown -R appuser:appuser /app/data /app/credentials

# entrypoint.sh に実行権限を付与
RUN chmod +x /app/backend/app/entrypoint.sh

# デバッグ情報の出力
RUN echo "--- Final Image Debug Info ---"
RUN ls -la /app
RUN ls -la /app/backend
RUN ls -la /opt/conda/bin
RUN echo "main.py: $(test -f /app/backend/main.py && echo OK || echo NG)"
RUN echo "entrypoint.sh: $(test -f /app/backend/app/entrypoint.sh && echo OK || echo NG)"
RUN echo "entrypoint.sh executable: $(test -x /app/backend/app/entrypoint.sh && echo YES || echo NO)"
RUN echo "Conda env path: $PATH"
RUN echo "Default Conda env: $CONDA_DEFAULT_ENV" # 設定していれば表示される
RUN echo "Active Conda env (should be base): $(conda info --envs | grep '*' | awk '{print $1}')"
RUN echo "Python version: $(python --version)"
RUN echo "pip freeze (first 5 lines):"
RUN python -m pip freeze | head -n 5
RUN echo "--- End Debug Info ---"

# ポートの公開
EXPOSE 8000

# 実行ユーザーを appuser に切り替え
USER appuser

# コンテナ起動時に実行されるコマンド (backend内のentrypoint.shを使用)
ENTRYPOINT ["/app/backend/app/entrypoint.sh"]