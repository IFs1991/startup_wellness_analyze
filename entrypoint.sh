#!/bin/bash
# スタートアップ分析プラットフォーム: バックエンドアプリケーション起動スクリプト
# メモリ最適化とGCP割引時間帯を活用するための設定を含む

set -e

# タイトルの表示
echo "=================================================="
echo "スタートアップウェルネス分析プラットフォーム - バックエンド"
echo "=================================================="

# 依存関係チェック
echo "必要な依存関係を確認しています..."
# SQLAlchemyがインストールされているか確認し、必要なら追加インストール
if ! python -c "import sqlalchemy" &> /dev/null; then
    echo "SQLAlchemyがインストールされていません。インストールします..."
    pip install sqlalchemy
fi

# PostgreSQLドライバがインストールされているか確認
if ! python -c "import psycopg2" &> /dev/null; then
    echo "psycopg2がインストールされていません。インストールします..."
    pip install psycopg2-binary
fi

# 環境変数の確認
APP_ENV=${APP_ENV:-production}
LOG_LEVEL=${LOG_LEVEL:-info}
WORKERS=${WORKERS:-4}

echo "実行環境: $APP_ENV"
echo "ログレベル: $LOG_LEVEL"
echo "ワーカー数: $WORKERS"

# ディレクトリ構造の確認
if [ ! -d "/app/data" ]; then
    echo "データディレクトリを作成します..."
    mkdir -p /app/data
fi

# パフォーマンス設定
echo "システムリソースを最適化しています..."

# メモリ最適化
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-100000}
export PYTHONMALLOC=${PYTHONMALLOC:-malloc}

# CPU使用の最適化
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}

# タイムゾーン設定（東京）
export TZ=${TZ:-"Asia/Tokyo"}

# GCP割引時間帯チェック
check_discount_hours() {
    # 現在時刻の取得
    current_hour=$(date +%H)
    day_of_week=$(date +%u)  # 1-7, 6と7が週末

    DISCOUNT_HOURS_START=${DISCOUNT_HOURS_START:-22}
    DISCOUNT_HOURS_END=${DISCOUNT_HOURS_END:-8}
    WEEKEND_DISCOUNT=${WEEKEND_DISCOUNT:-true}

    is_weekend=false
    if [ "$day_of_week" -ge 6 ]; then
        is_weekend=true
    fi

    is_discount_hour=false
    if [ "$current_hour" -ge "$DISCOUNT_HOURS_START" ] || [ "$current_hour" -lt "$DISCOUNT_HOURS_END" ]; then
        is_discount_hour=true
    fi

    if [ "$is_weekend" = true ] && [ "$WEEKEND_DISCOUNT" = true ]; then
        echo "現在は週末のため、GCP割引時間帯です。リソースを最大限活用できます。"
        return 0
    elif [ "$is_discount_hour" = true ]; then
        echo "現在はGCP割引時間帯（$DISCOUNT_HOURS_START:00-$DISCOUNT_HOURS_END:00）です。リソースを最大限活用できます。"
        return 0
    else
        echo "現在はGCP割引時間帯ではありません。リソース使用量を最適化します。"
        # 非割引時間帯ではワーカー数を減らす
        WORKERS=$((WORKERS / 2))
        if [ "$WORKERS" -lt 1 ]; then
            WORKERS=1
        fi
        echo "ワーカー数を $WORKERS に調整しました"
        return 1
    fi
}

# 健全性チェック用エンドポイント確保
ensure_health_endpoint() {
    # main.pyファイルが存在するか確認
    if [ -f "main.py" ]; then
        # ファイル内に/healthエンドポイントの定義がない場合は追加
        if ! grep -q "@app.get(\"/health\")" main.py; then
            echo "健全性チェック用エンドポイントを追加します..."
            echo '
import datetime

@app.get("/health")
def health_check():
    """コンテナの健全性チェック用エンドポイント"""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}
' >> main.py
        fi
    fi
}

# メモリ使用量をモニタリングする関数
start_memory_monitor() {
    if [ "$APP_ENV" = "production" ]; then
        # 別のプロセスでメモリモニタリングを開始
        (
            while true; do
                # 現在のメモリ使用量を取得
                memory_usage=$(ps -o rss= -p $$)
                memory_usage_mb=$((memory_usage / 1024))

                # 過去のピーク値と比較
                if [ -f "/app/data/peak_memory.txt" ]; then
                    peak_memory=$(cat /app/data/peak_memory.txt)
                else
                    peak_memory=0
                fi

                # 新しいピーク値を記録
                if [ "$memory_usage_mb" -gt "$peak_memory" ]; then
                    echo "$memory_usage_mb" > /app/data/peak_memory.txt
                fi

                # ロギング（1時間に1回）
                if [ $(($(date +%M) % 60)) -eq 0 ] && [ "$(date +%S)" -eq 0 ]; then
                    echo "メモリ使用量: ${memory_usage_mb}MB (ピーク: $(cat /app/data/peak_memory.txt)MB)"
                fi

                sleep 60
            done
        ) &
    fi
}

# GCP割引時間帯チェック
check_discount_hours

# 健全性チェックエンドポイントの確保
ensure_health_endpoint

# メモリモニタリングの開始
start_memory_monitor

# 前処理（必要に応じて）
if [ -f "/app/scripts/preload.py" ]; then
    echo "前処理スクリプトを実行しています..."
    python /app/scripts/preload.py
fi

# GCPコスト最適化スクリプトの実行（存在する場合）
if [ -f "/app/scripts/optimize_gcp_costs.py" ]; then
    echo "GCPコスト最適化スクリプトを実行しています..."
    python /app/scripts/optimize_gcp_costs.py --check-only
fi

# アプリケーション起動の前にメッセージを表示
echo "バックエンドサーバーを起動しています..."
echo "アクセスURL: http://localhost:8000"
echo "--------------------------------------------------"

# 環境変数に基づいて適切なコマンドを実行
if [ "$APP_ENV" = "development" ]; then
    # 開発モード：ホットリロードを有効化
    exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level "$LOG_LEVEL"
else
    # 本番モード：マルチワーカーで起動
    exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers "$WORKERS" --log-level "$LOG_LEVEL"
fi
