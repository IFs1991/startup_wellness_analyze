#!/bin/sh
# シンプルなヘルスチェックスクリプト
# Next.jsアプリケーションが正常に動作しているか確認

# アプリケーションのステータスを確認
if curl -s -f http://localhost:3000/ > /dev/null; then
    echo "Healthcheck: OK"
    exit 0
else
    echo "Healthcheck: FAILED"
    exit 1
fi