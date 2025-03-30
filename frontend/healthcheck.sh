#!/bin/sh
# シンプルなヘルスチェックスクリプト
# Nginxが正常に動作しているか確認

# Nginxのステータスを確認
if curl -s -f http://localhost/ > /dev/null; then
    echo "Healthcheck: OK"
    exit 0
else
    echo "Healthcheck: FAILED"
    exit 1
fi