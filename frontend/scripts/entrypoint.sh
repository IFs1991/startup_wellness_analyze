#!/bin/sh

# エラー発生時に終了
set -e

# デバッグモードで実行（詳細なログを表示）
if [ "$DEBUG" = "true" ]; then
  set -x
fi

# 環境変数のチェック
if [ -z "$VITE_API_URL" ]; then
  echo "警告: VITE_API_URL環境変数が設定されていません。デフォルト値を使用します。"
  export VITE_API_URL="http://backend:8000/api"
fi

# Firebase環境変数のチェック
if [ -z "$VITE_FIREBASE_API_KEY" ] || [ -z "$VITE_FIREBASE_PROJECT_ID" ]; then
  echo "注意: Firebase環境変数が設定されていない可能性があります。認証機能が正常に動作しない可能性があります。"
fi

# システム情報の表示
echo "フロントエンドサーバー起動: $(date)"
echo "- APIエンドポイント: $VITE_API_URL"
echo "- ホスト名: $(hostname)"

# バックエンドのヘルスチェックをバックグラウンドで実行
echo "バックエンドの接続確認をバックグラウンドで実行中..."
(
  BACKEND_URL=$(echo $VITE_API_URL | sed 's/\/api$/\/health/')
  MAX_RETRIES=5
  RETRY_COUNT=0
  RETRY_DELAY=3

  while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f $BACKEND_URL > /dev/null 2>&1; then
      echo "バックエンドの接続確認: 成功"
      exit 0
    else
      echo "バックエンドの接続確認: 失敗 (リトライ $((RETRY_COUNT+1))/$MAX_RETRIES)"
      RETRY_COUNT=$((RETRY_COUNT+1))
      [ $RETRY_COUNT -eq $MAX_RETRIES ] || sleep $RETRY_DELAY
    fi
  done

  echo "警告: バックエンドに接続できません。APIリクエストはフォールバックレスポンスを返します。"
) &

# Nginxサーバーを起動（メインプロセス）
echo "Nginxサーバーを起動しています..."
exec nginx -g "daemon off;"