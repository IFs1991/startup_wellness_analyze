#!/bin/bash
# スタートアップ分析プラットフォーム: バックエンドアプリケーション起動スクリプト
# (修正版: 動的処理を削除、conda runを使用)
# メモリ最適化とGCP割引時間帯を活用するための設定を含む

set -e

# タイトルの表示
echo "=================================================="
echo "スタートアップウェルネス分析プラットフォーム - バックエンド"
echo "=================================================="

# main.pyの存在確認
# Note: DockerfileのCOPY命令により /app/backend/main.py に配置される想定
if [ ! -f "/app/backend/main.py" ]; then
    echo "エラー: /app/backend/main.pyが見つかりません。アプリケーションを起動できません。"
    exit 1
fi

# 環境変数の確認 (警告のみ)
if [ -z "$GEMINI_API_KEY" ]; then
    echo "警告: GEMINI_API_KEY環境変数が設定されていません。"
fi
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ] || [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "警告: GOOGLE_APPLICATION_CREDENTIALSが正しく設定されていないか、ファイルが存在しません。"
fi

# Condaの起動を確認と初期化 (念のため実行)
echo "Conda環境を確認・初期化しています..."
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    . "/opt/conda/etc/profile.d/conda.sh"
else
    echo "警告: Conda初期化スクリプトが見つかりません。"
fi

# Conda環境名
CONDA_ENV_NAME="startup_wellness_analyze" # environment.yml と合わせる

# 環境変数の設定
APP_ENV=${APP_ENV:-production}
LOG_LEVEL=${LOG_LEVEL:-info}
WORKERS_RAW=${WORKERS:-4} # 元の変数を保持

# WORKERS変数からコメントを除去し、数値のみを抽出
WORKERS=$(echo "$WORKERS_RAW" | sed 's/#.*//' | xargs)
# 数値でない場合はデフォルト値4を設定
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]]; then
    echo "警告: WORKERS環境変数が数値ではありません。デフォルト値4を使用します。"
    WORKERS=4
fi

echo "実行環境: $APP_ENV"
echo "ログレベル: $LOG_LEVEL"
echo "ワーカー数 (初期値): $WORKERS" # 抽出後の値を表示

# ディレクトリ構造の確認
if [ ! -d "/app/data" ]; then
    echo "データディレクトリ /app/data を作成します..."
    mkdir -p /app/data
fi

# パフォーマンス設定
echo "システムリソース設定を適用しています..."
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-100000}
export PYTHONMALLOC=${PYTHONMALLOC:-malloc}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export TZ=${TZ:-"Asia/Tokyo"}

# GCP割引時間帯チェック関数
check_discount_hours() {
    current_hour=$(date +%H)
    day_of_week=$(date +%u)
    DISCOUNT_HOURS_START=${DISCOUNT_HOURS_START:-22}
    DISCOUNT_HOURS_END=${DISCOUNT_HOURS_END:-8}
    WEEKEND_DISCOUNT=${WEEKEND_DISCOUNT:-true}
    is_weekend=false
    if [ "$day_of_week" -ge 6 ]; then is_weekend=true; fi
    is_discount_hour=false
    if [ "$current_hour" -ge "$DISCOUNT_HOURS_START" ] || [ "$current_hour" -lt "$DISCOUNT_HOURS_END" ]; then is_discount_hour=true; fi

    if [ "$is_weekend" = true ] && [ "$WEEKEND_DISCOUNT" = true ]; then
        echo "現在は週末のため、GCP割引時間帯です。"
    elif [ "$is_discount_hour" = true ]; then
        echo "現在はGCP割引時間帯（$DISCOUNT_HOURS_START:00-$DISCOUNT_HOURS_END:00）です。"
    else
        echo "現在はGCP割引時間帯ではありません。ワーカー数を調整します。"
        local original_workers=$WORKERS
        WORKERS=$((WORKERS / 2))
        if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
        if [ "$original_workers" != "$WORKERS" ]; then echo "ワーカー数を $WORKERS に調整しました"; fi
    fi
}

# メモリ使用量をモニタリングする関数
start_memory_monitor() {
    if [ "$APP_ENV" = "production" ]; then
        echo "メモリモニタリングをバックグラウンドで開始します (1時間ごとにログ出力)..."
        (
            while true; do
                # psコマンドが見つからない場合を考慮
                local memory_usage=$(ps -o rss= -p $$ 2>/dev/null || echo 0)
                local memory_usage_mb=0
                if [[ "$memory_usage" =~ ^[0-9]+$ ]]; then
                    memory_usage_mb=$((memory_usage / 1024))
                fi

                local peak_file="/app/data/peak_memory.txt"
                local peak_memory=0
                if [ -f "$peak_file" ]; then
                     # ファイル内容が数値か確認
                     local content=$(cat "$peak_file")
                     if [[ "$content" =~ ^[0-9]+$ ]]; then
                         peak_memory=$content
                     fi
                fi

                # 数値として比較
                if [[ "$memory_usage_mb" -gt "$peak_memory" ]]; then
                    echo "$memory_usage_mb" > "$peak_file"
                    peak_memory=$memory_usage_mb
                fi

                if [ "$(date +%M%S)" = "0000" ]; then
                    echo "[Memory Monitor] Usage: ${memory_usage_mb}MB (Peak: ${peak_memory}MB)"
                fi
                sleep 60
            done
        ) &
    fi
}

# --- 実行セクション ---

# GCP割引時間帯チェックとワーカー数調整
check_discount_hours
echo "最終的なワーカー数: $WORKERS"

# メモリモニタリングの開始 (本番環境のみ)
start_memory_monitor

# 前処理（必要に応じて）
if [ -f "/app/scripts/preload.py" ]; then
    echo "前処理スクリプト /app/scripts/preload.py を実行しています..."
    # conda run で実行
    conda run -n "$CONDA_ENV_NAME" python /app/scripts/preload.py
fi

# GCPコスト最適化スクリプトの実行（存在する場合）
if [ -f "/app/scripts/optimize_gcp_costs.py" ]; then
    echo "GCPコスト最適化スクリプト /app/scripts/optimize_gcp_costs.py を実行しています..."
    # conda run で実行
    conda run -n "$CONDA_ENV_NAME" python /app/scripts/optimize_gcp_costs.py --check-only
fi

# アプリケーション起動
echo "バックエンドサーバー (Uvicorn) を起動しています..."
echo "アクセスURL (コンテナ内部): http://0.0.0.0:8000"
echo "--------------------------------------------------"

# 起動コマンドの決定と実行 (conda runを使用, モジュールパスを修正)
# main.py は /app/backend/main.py にあるため、backend.main:app と指定
UVICORN_COMMON_ARGS="backend.main:app --host 0.0.0.0 --port 8000 --log-level $LOG_LEVEL"

if [ "$APP_ENV" = "development" ]; then
    # 開発モード (ホットリロードが必要な場合は --reload を追加)
    echo "開発モードで起動します..."
    exec conda run -n "$CONDA_ENV_NAME" uvicorn $UVICORN_COMMON_ARGS # --reload
else
    # 本番モード：マルチワーカーで起動
    echo "本番モードで起動します (ワーカー数: $WORKERS)..."
    # 本番用に追加された可能性のある引数を維持
    exec conda run -n "$CONDA_ENV_NAME" uvicorn $UVICORN_COMMON_ARGS --workers "$WORKERS" --proxy-headers --forwarded-allow-ips '*' --limit-concurrency 4 --backlog 2048 --timeout-keep-alive 5
fi