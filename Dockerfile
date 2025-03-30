FROM condaforge/mambaforge:latest

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
COPY environment.yml .
RUN mamba env update -n base -f environment.yml && \
    mamba clean -afy && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.pyc' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete

# スクリプトやその他のファイルをコピー
COPY ./backend/ /app/backend/
COPY ./scripts/ /app/scripts/
COPY ./entrypoint.sh /app/

# entrypoint.shに実行権限を付与
RUN chmod +x /app/entrypoint.sh

# 一般ユーザーで実行するための設定
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# ポートの公開
EXPOSE 8000

# メモリ使用量監視コマンドの作成
RUN echo '#!/bin/bash\nps -o pid,user,%mem,command ax | sort -b -k3 -r' > /app/scripts/memory_usage.sh && \
    chmod +x /app/scripts/memory_usage.sh

# デフォルトのコマンド
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]