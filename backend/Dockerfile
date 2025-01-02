FROM python:3.9

WORKDIR /app

# 依存関係のインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 開発用依存関係のインストール（オプション）
COPY requirements-dev.txt ./
RUN if [ "$ENVIRONMENT" = "development" ] ; then pip install --no-cache-dir -r requirements-dev.txt ; fi

# アプリケーションのコピー
COPY . .

# Pythonパスの設定
ENV PYTHONPATH=/app:/app/backend
ENV PYTHONUNBUFFERED=1

# ポートの設定
EXPOSE 8000

# 起動コマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]