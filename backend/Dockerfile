# ビルドステージ: 依存関係のインストールと環境構築を行います
FROM python:3.12-slim-bookworm as builder

# ビルド環境の最適化設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app" \
    GOOGLE_APPLICATION_CREDENTIALS="/app/credentials/startupwellnessanalyze-445505-firebase-adminsdk-2hbn3-dc3bee234b.json"

WORKDIR /app

# ビルドに必要な基本ツールのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    g++ \
    gcc \
    libblas-dev \
    liblapack-dev \
    libopenblas0 \
    gfortran \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 依存関係の管理ファイルをコピー
COPY requirements.txt .

# 仮想環境の作成と依存関係のインストール
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -U pip==24.0 && \
    pip install --no-cache-dir -r requirements.txt

# 本番環境ステージ
FROM python:3.12-slim-bookworm as runtime

# 環境変数の設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    GOOGLE_APPLICATION_CREDENTIALS="/app/credentials/startupwellnessanalyze-445505-firebase-adminsdk-2hbn3-dc3bee234b.json" \
    PYTHONASYNCIODEBUG=1 \
    REQUESTS_TIMEOUT=30 \
    GOOGLE_API_TIMEOUT=60 \
    # BLAS設定
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    ENVIRONMENT=development

WORKDIR /app

# 必要なパッケージのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    g++ \
    libblas3 \
    liblapack3 \
    libpq5 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* && \
    useradd -m -s /bin/bash app && \
    mkdir -p /app /var/log/app /app/credentials /home/app/.cache/pip && \
    chown -R app:app /app /var/log/app /home/app && \
    chmod -R 755 /home/app/.cache && \
    chmod 750 /app/credentials && \
    mkdir -p /app/logs && chmod 777 /app/logs

# 仮想環境のコピー
COPY --from=builder /opt/venv /opt/venv

# アプリケーションファイルのコピー
COPY --chown=app:app . .

# Firebase認証情報ファイルのコピー
COPY --chown=app:app ./credentials/startupwellnessanalyze-445505-firebase-adminsdk-2hbn3-dc3bee234b.json /app/backend/credentials/

# 起動スクリプトの作成
RUN printf '#!/bin/bash\n\
cd /app\n\
echo "Python search path:"\n\
python -c "import sys; print(\"\n\".join(sys.path))"\n\
echo -e "\nInstalled packages:"\n\
pip list\n\
echo -e "\nStarting application..."\n\
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --limit-max-requests 10000 --timeout-keep-alive 75\n' > /app/start.sh && \
    chmod +x /app/start.sh && \
    chown app:app /app/start.sh

USER app

# ヘルスチェックの改善
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/app/start.sh"]