FROM python:3.9

WORKDIR /app

# 依存関係のインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-multipart==0.0.9 && \
    pip install --no-cache-dir pandas==2.2.2

# 開発用依存関係のインストール（オプション）
COPY requirements-dev.txt ./
RUN if [ "$ENVIRONMENT" = "development" ] ; then pip install --no-cache-dir -r requirements-dev.txt ; fi

# アプリケーションのコピー
COPY . .

# パッケージのインストール
ENV PYTHONPATH=/app:${PYTHONPATH}
RUN pip install -e .

# ログ設定
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV ENVIRONMENT=production
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/startupwellnessanalyze-445505-6a7cc0e46cac.json

# Firebase認証情報のマウントポイント
VOLUME /app/credentials

# ポートの設定
EXPOSE 8000

# 起動コマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "4", "--log-level", "info"]