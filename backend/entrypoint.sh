#!/bin/bash
# スタートアップ分析プラットフォーム: バックエンドアプリケーション起動スクリプト
# メモリ最適化とGCP割引時間帯を活用するための設定を含む

set -e

# タイトルの表示
echo "=================================================="
echo "スタートアップウェルネス分析プラットフォーム - バックエンド"
echo "=================================================="

# main.pyの存在確認
if [ ! -f "/app/main.py" ]; then
    echo "エラー: main.pyが見つかりません。アプリケーションを起動できません。"
    echo "ディレクトリ構造を確認してください。"
    exit 1
fi

# 環境変数の確認
if [ -z "$GEMINI_API_KEY" ]; then
    echo "警告: GEMINI_API_KEY環境変数が設定されていません。"
    echo "本番環境では正しく設定してください。"
fi

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ] || [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "警告: GOOGLE_APPLICATION_CREDENTIALS環境変数が正しく設定されていないか、ファイルが存在しません。"
    echo "Firebase/Firestoreの機能が正常に動作しない可能性があります。"
fi

# Condaの起動を確認
echo "Conda環境を確認しています..."
if ! command -v conda &> /dev/null; then
    echo "エラー: condaコマンドが見つかりません。PATH環境変数が正しく設定されているか確認してください。"
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        echo "Condaシェルスクリプトを読み込みます..."
        . "/opt/conda/etc/profile.d/conda.sh"
    fi
fi

# conda init 相当の処理を実行
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    echo "Condaシェルスクリプトを初期化します..."
    . "/opt/conda/etc/profile.d/conda.sh"
    USE_CONDA_RUN=false
else
    echo "Condaシェルスクリプトが見つかりません。直接conda runを使用します。"
    USE_CONDA_RUN=true
fi

# Conda環境をアクティブ化 (シェルスクリプトが初期化されている場合のみ)
if [ "$USE_CONDA_RUN" = false ]; then
    echo "Conda環境をアクティブ化しています: startup_wellness_analyze"
    if ! conda activate startup_wellness_analyze; then
        echo "警告: conda activateに失敗しました。直接conda runを使用します。"
        USE_CONDA_RUN=true
    else
        echo "Conda環境が正常にアクティブ化されました。"
    fi
fi

# ルートディレクトリの.envファイルへのシンボリックリンクを作成
if [ -f "/app/../.env" ] && [ ! -f "/app/.env" ]; then
    echo "ルートの.envファイルへのシンボリックリンクを作成します..."
    ln -sf /app/../.env /app/.env
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
    if [ "$USE_CONDA_RUN" = true ]; then
        conda run -n startup_wellness_analyze python /app/scripts/preload.py
    else
        python /app/scripts/preload.py
    fi
fi

# GCPコスト最適化スクリプトの実行（存在する場合）
if [ -f "/app/scripts/optimize_gcp_costs.py" ]; then
    echo "GCPコスト最適化スクリプトを実行しています..."
    if [ "$USE_CONDA_RUN" = true ]; then
        conda run -n startup_wellness_analyze python /app/scripts/optimize_gcp_costs.py --check-only
    else
        python /app/scripts/optimize_gcp_costs.py --check-only
    fi
fi

# アプリケーション起動の前にメッセージを表示
echo "バックエンドサーバーを起動しています..."

# 必要なパッケージのチェックとインストール
echo "必要なパッケージの確認を行います..."

# FastAPI関連パッケージのチェック
if ! python -c "import fastapi" &> /dev/null; then
    echo "fastapi が見つかりません。インストールを試みます..."
    pip install fastapi==0.110.0
fi

if ! python -c "import uvicorn" &> /dev/null; then
    echo "uvicorn が見つかりません。インストールを試みます..."
    pip install uvicorn==0.27.1
fi

# データベース関連パッケージのチェック
if ! python -c "import sqlalchemy" &> /dev/null; then
    echo "sqlalchemy が見つかりません。インストールを試みます..."
    pip install sqlalchemy==2.0.27
fi

if ! python -c "import psycopg2" &> /dev/null; then
    echo "psycopg2-binary が見つかりません。インストールを試みます..."
    pip install psycopg2-binary==2.9.9
fi

# Firebase関連パッケージのチェック
if ! python -c "import firebase_admin" &> /dev/null; then
    echo "firebase_admin が見つかりません。インストールを試みます..."
    pip install firebase-admin==6.2.0
fi

if ! python -c "import google.cloud.firestore" &> /dev/null; then
    echo "google-cloud-firestore が見つかりません。インストールを試みます..."
    pip install google-cloud-firestore==2.13.1
fi

# MFA関連パッケージのチェック
if ! python -c "import pyotp" &> /dev/null; then
    echo "pyotp が見つかりません。インストールを試みます..."
    pip install pyotp==2.8.0
fi

if ! python -c "import qrcode" &> /dev/null; then
    echo "qrcode が見つかりません。インストールを試みます..."
    pip install qrcode==7.4.2
fi

# Google Cloud関連パッケージのチェック
if ! python -c "import google.cloud.secret_manager" &> /dev/null; then
    echo "google-cloud-secret-manager が見つかりません。インストールを試みます..."
    pip install google-cloud-secret-manager==2.16.3
fi

if ! python -c "import google.cloud.storage" &> /dev/null; then
    echo "google-cloud-storage が見つかりません。インストールを試みます..."
    pip install google-cloud-storage==2.12.0
fi

if ! python -c "import google.cloud.bigquery" &> /dev/null; then
    echo "google-cloud-bigquery が見つかりません。インストールを試みます..."
    pip install google-cloud-bigquery==3.12.0
fi

# Web関連パッケージのチェック
if ! python -c "import jose" &> /dev/null; then
    echo "python-joseが見つかりません。インストールを試みます..."
    pip install python-jose[cryptography]==3.3.0
fi

if ! python -c "import multipart" &> /dev/null; then
    echo "python-multipartが見つかりません。インストールを試みます..."
    pip install python-multipart==0.0.6
fi

# アルゴリズム関連パッケージのチェック
if ! python -c "import pandas" &> /dev/null; then
    echo "pandasが見つかりません。インストールを試みます..."
    pip install pandas==2.1.0
fi

if ! python -c "import numpy" &> /dev/null; then
    echo "numpyが見つかりません。インストールを試みます..."
    pip install numpy==1.24.3
fi

# パスワードハッシュ関連パッケージのチェック
if ! python -c "import argon2" &> /dev/null; then
    echo "argon2-cffiが見つかりません。インストールを試みます..."
    pip install argon2-cffi==23.1.0
fi

# NoSQLデータベース関連パッケージのチェック
if ! python -c "import neo4j" &> /dev/null; then
    echo "neo4jが見つかりません。インストールを試みます..."
    pip install neo4j==5.14.0
fi

# キャッシュ関連パッケージのチェック
if ! python -c "import redis" &> /dev/null; then
    echo "redisが見つかりません。インストールを試みます..."
    pip install redis==5.0.1
fi

# パスワードハッシュライブラリのチェック
if ! python -c "import passlib" &> /dev/null; then
    echo "passlibが見つかりません。インストールを試みます..."
    pip install passlib==1.7.4
fi

# 暗号化ライブラリのチェック
if ! python -c "import Crypto" &> /dev/null; then
    echo "pycryptodomeが見つかりません。インストールを試みます..."
    pip install pycryptodome==3.19.0
fi

# データ分析関連パッケージのチェック
if ! python -c "import statsmodels" &> /dev/null; then
    echo "statsmodelsが見つかりません。インストールを試みます..."
    pip install statsmodels==0.14.0
fi

if ! python -c "import scipy" &> /dev/null; then
    echo "scipyが見つかりません。インストールを試みます..."
    pip install scipy==1.10.1
fi

if ! python -c "import sklearn" &> /dev/null; then
    echo "scikit-learnが見つかりません。インストールを試みます..."
    pip install scikit-learn==1.3.0
fi

if ! python -c "import matplotlib" &> /dev/null; then
    echo "matplotlibが見つかりません。インストールを試みます..."
    pip install matplotlib==3.8.0
fi

if ! python -c "import seaborn" &> /dev/null; then
    echo "seabornが見つかりません。インストールを試みます..."
    pip install seaborn==0.12.2
fi

if ! python -c "import lifelines" &> /dev/null; then
    echo "lifelinesが見つかりません。インストールを試みます..."
    pip install lifelines==0.27.4
fi

# データ処理・可視化関連
if ! python -c "import plotly" &> /dev/null; then
    echo "plotlyが見つかりません。インストールを試みます..."
    pip install plotly==5.15.0
fi

if ! python -c "import pyarrow" &> /dev/null; then
    echo "pyarrowが見つかりません。インストールを試みます..."
    pip install pyarrow==12.0.1
fi

if ! python -c "import dask" &> /dev/null; then
    echo "daskが見つかりません。インストールを試みます..."
    pip install dask==2023.7.1
fi

# 時系列分析
if ! python -c "import prophet" &> /dev/null; then
    echo "prophetが見つかりません。インストールを試みます..."
    pip install prophet==1.1.4
fi

# 連合学習
if ! python -c "import flwr" &> /dev/null; then
    echo "flwrが見つかりません。インストールを試みます..."
    pip install flwr==1.16.0
fi

# protobufの互換性確保
if ! python -c "import google.protobuf" &> /dev/null || python -c "import sys; from google.protobuf import __version__ as v; sys.exit(0 if v.startswith('3.') else 1)" &> /dev/null; then
    echo "protobufをflwrと互換性のあるバージョンに設定します..."
    pip install protobuf==3.20.3
fi

# NLP関連
if ! python -c "import nltk" &> /dev/null; then
    echo "nltkが見つかりません。インストールを試みます..."
    pip install nltk==3.8.1
fi

# HTTPリクエスト
if ! python -c "import requests" &> /dev/null; then
    echo "requestsが見つかりません。インストールを試みます..."
    pip install requests==2.31.0
fi

# 暗号化
if ! python -c "import cryptography" &> /dev/null; then
    echo "cryptographyが見つかりません。インストールを試みます..."
    pip install cryptography==41.0.3
fi

# ウェブフレームワーク関連
if ! python -c "import dash" &> /dev/null; then
    echo "dashが見つかりません。インストールを試みます..."
    pip install dash==2.9.3
fi

# モニタリングとロギング
if ! python -c "import prometheus_client" &> /dev/null; then
    echo "prometheus-clientが見つかりません。インストールを試みます..."
    pip install prometheus-client==0.17.1
fi

if ! python -c "import loguru" &> /dev/null; then
    echo "loguruが見つかりません。インストールを試みます..."
    pip install loguru==0.7.0
fi

# 環境変数管理
if ! python -c "import dotenv" &> /dev/null; then
    echo "python-dotenvが見つかりません。インストールを試みます..."
    pip install python-dotenv==1.0.0
fi

# YAML処理
if ! python -c "import yaml" &> /dev/null; then
    echo "PyYAMLが見つかりません。インストールを試みます..."
    pip install PyYAML==6.0
fi

# システムユーティリティ
if ! python -c "import psutil" &> /dev/null; then
    echo "psutilが見つかりません。インストールを試みます..."
    pip install psutil==5.9.5
fi

# 並列処理・最適化
if ! python -c "import numba" &> /dev/null; then
    echo "numbaが見つかりません。インストールを試みます..."
    pip install numba==0.57.1
fi

# クラウドストレージアクセス
if ! python -c "import s3fs" &> /dev/null; then
    echo "s3fsが見つかりません。インストールを試みます..."
    pip install s3fs==2023.6.0
fi

if ! python -c "import gcsfs" &> /dev/null; then
    echo "gcsfsが見つかりません。インストールを試みます..."
    pip install gcsfs==2023.6.0
fi

# 因果推論
if ! python -c "import causalimpact" &> /dev/null; then
    echo "causalimpactが見つかりません。インストールを試みます..."
    pip install causalimpact==0.2.6
fi

if ! python -c "import dowhy" &> /dev/null; then
    echo "dowhyが見つかりません。インストールを試みます..."
    pip install dowhy==0.10.0
fi

# その他の解析ライブラリ
if ! python -c "import econml" &> /dev/null; then
    echo "econmlが見つかりません。インストールを試みます..."
    pip install econml==0.14.1
fi

if ! python -c "import mlxtend" &> /dev/null; then
    echo "mlxtendが見つかりません。インストールを試みます..."
    pip install mlxtend==0.22.0
fi

# データバリデーション
if ! python -c "import great_expectations" &> /dev/null; then
    echo "great_expectationsが見つかりません。インストールを試みます..."
    pip install great_expectations==0.17.19
fi

# レポート生成
if ! python -c "import reportlab" &> /dev/null; then
    echo "reportlabが見つかりません。インストールを試みます..."
    pip install reportlab==4.0.4
fi

if ! python -c "import jinja2" &> /dev/null; then
    echo "jinja2が見つかりません。インストールを試みます..."
    pip install jinja2==3.1.2
fi

if ! python -c "import weasyprint" &> /dev/null; then
    echo "weasyprintが見つかりません。インストールを試みます..."
    pip install weasyprint==59.0
fi

# ウェブスクレイピング
if ! python -c "import bs4" &> /dev/null; then
    echo "beautifulsoup4が見つかりません。インストールを試みます..."
    pip install beautifulsoup4==4.12.2
fi

if ! python -c "import httpx" &> /dev/null; then
    echo "httpxが見つかりません。インストールを試みます..."
    pip install httpx==0.24.1
fi

# 認証・セキュリティ
if ! python -c "import bcrypt" &> /dev/null; then
    echo "bcryptが見つかりません。インストールを試みます..."
    pip install bcrypt==4.0.1
fi

if ! python -c "import authlib" &> /dev/null; then
    echo "authlibが見つかりません。インストールを試みます..."
    pip install authlib==1.2.1
fi

# Google API
if ! python -c "import googleapiclient" &> /dev/null; then
    echo "google-api-python-clientが見つかりません。インストールを試みます..."
    pip install google-api-python-client==2.95.0
fi

if ! python -c "import google.generativeai" &> /dev/null; then
    echo "google-generativeaiが見つかりません。インストールを試みます..."
    pip install google-generativeai==0.3.1
fi

# バックグラウンド処理
if ! python -c "import celery" &> /dev/null; then
    echo "celeryが見つかりません。インストールを試みます..."
    pip install celery==5.4.0
fi

# メール検証ライブラリがインストールされているか確認
python -c "import email_validator" || pip install email-validator
# Pydanticのメール検証拡張がインストールされているか確認
python -c "import pydantic" && pip install pydantic[email]

echo "必要なパッケージのインストールを試みました。"

# アクセスURL: http://localhost:8000
echo "--------------------------------------------------"

# conda環境が存在しているかチェック
if [ "$USE_CONDA_RUN" = true ]; then
    echo "Conda run を使用して起動します..."
    if [ "$APP_ENV" = "development" ]; then
        # 開発モード：ホットリロードを有効化
        exec conda run -n startup_wellness_analyze --no-capture-output uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level "$LOG_LEVEL"
    else
        # 本番モード：マルチワーカーで起動
        exec conda run -n startup_wellness_analyze --no-capture-output uvicorn main:app --host 0.0.0.0 --port 8000 --workers "$WORKERS" --log-level "$LOG_LEVEL" --limit-concurrency 4 --backlog 2048 --timeout-keep-alive 5 --proxy-headers --forwarded-allow-ips '*'
    fi
else
    echo "直接コマンドを実行します..."
    if [ "$APP_ENV" = "development" ]; then
        # 開発モード：ホットリロードを有効化
        exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level "$LOG_LEVEL"
    else
        # 本番モード：マルチワーカーで起動
        exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers "$WORKERS" --log-level "$LOG_LEVEL" --limit-concurrency 4 --backlog 2048 --timeout-keep-alive 5 --proxy-headers --forwarded-allow-ips '*'
    fi
fi