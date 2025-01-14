# ビルドステージ
FROM python:3.9-slim-buster AS builder

# ビルド環境の設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 必要なビルドツールのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 依存関係のインストール
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# 本番環境ステージ
FROM python:3.9-slim-buster

# 環境変数の設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

WORKDIR /app

# 仮想環境のコピー
COPY --from=builder /opt/venv /opt/venv

# セキュリティ設定
RUN useradd -r -s /bin/false app && \
    mkdir -p /app /var/log/app && \
    chown -R app:app /app /var/log/app

# アプリケーションファイルのコピー
COPY --chown=app:app . .

# 実行ユーザーの変更
USER app

# ポート設定
EXPOSE 8000

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 起動コマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]