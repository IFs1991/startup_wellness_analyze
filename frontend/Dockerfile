# ベースイメージとしてNode.js 22 Alpineを使用
FROM node:22-alpine AS build

# アプリケーションのビルド環境をセットアップ
WORKDIR /app

# ビルド時の環境変数を設定
ENV NODE_ENV=development \
    VITE_API_URL=http://localhost:8000

# 必要な開発ツールをインストール
RUN apk add --no-cache python3 make g++

# npmキャッシュの設定を最適化
RUN npm config set cache /app/.npm-cache

# 依存関係ファイルをコピー
COPY package*.json ./

# 開発モードで依存関係をインストール
RUN npm ci --legacy-peer-deps && \
    npm install -D @types/node && \
    # 不要なキャッシュを削除してイメージサイズを削減
    npm cache clean --force

# プロジェクトのソースコードをコピー
COPY . .

# TypeScriptの型チェックとプロダクションビルドを実行
RUN npm run type-check && \
    NODE_ENV=production npm run build

# 本番環境用のステージ
FROM nginx:alpine

# Nginxの設定ファイルをコピー
COPY nginx.conf /etc/nginx/conf.d/default.conf

# ビルド成果物をNginxの公開ディレクトリにコピー
COPY --from=build /app/dist /usr/share/nginx/html

# セキュリティとパーミッションの設定
RUN chmod -R 755 /usr/share/nginx/html && \
    chown -R nginx:nginx /usr/share/nginx/html /var/cache/nginx /var/log/nginx && \
    # Nginxの基本設定を最適化
    sed -i 's/user  nginx/user nginx/g' /etc/nginx/nginx.conf && \
    sed -i 's/worker_processes  auto/worker_processes auto/g' /etc/nginx/nginx.conf && \
    sed -i 's/# server_tokens off/server_tokens off/g' /etc/nginx/nginx.conf && \
    # セキュリティヘッダーを追加
    echo "add_header X-Frame-Options DENY;" >> /etc/nginx/conf.d/default.conf && \
    echo "add_header X-Content-Type-Options nosniff;" >> /etc/nginx/conf.d/default.conf && \
    echo "add_header X-XSS-Protection \"1; mode=block\";" >> /etc/nginx/conf.d/default.conf

# 不要なパッケージとキャッシュを削除
RUN rm -rf /var/cache/apk/ /tmp/*

# Nginxがリッスンするポートを設定
EXPOSE 80
ENV PORT=80

# セキュリティ強化のためにnginxユーザーに切り替え
USER nginx

# ヘルスチェックの設定
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:${PORT}/ || exit 1

# Nginxをフォアグラウンドで実行
CMD ["nginx", "-g", "daemon off;"]