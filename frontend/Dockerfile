# ベースイメージとしてNode.js 22.13.1 Alpineを使用
FROM node:22.13.1-alpine AS build

# アプリケーションのビルド環境をセットアップ
WORKDIR /app

# ビルド時の環境変数を設定
ENV NODE_ENV=development \
    VITE_API_URL=http://localhost:8000/api

# 必要な開発ツールをインストール
RUN apk add --no-cache python3 make g++

# npmを最新の安定版に更新
RUN npm install -g npm@10.9.2

# npmキャッシュの設定を最適化
RUN npm config set cache /app/.npm-cache

# 必要なディレクトリを作成
RUN mkdir -p /app/src

# 依存関係ファイルをコピー
COPY package.json ./
COPY package-lock.json ./

# 設定ファイルをコピー
COPY tsconfig*.json ./
COPY vite.config.ts ./
COPY postcss.config.js ./
COPY tailwind.config.js ./
COPY index.html ./

# Vite環境定義ファイルをコピー
COPY src/vite-env.d.ts ./src/

# 依存関係のインストール
RUN if [ -f package-lock.json ]; then \
        npm ci; \
    else \
        npm install; \
    fi && \
    npm cache clean --force

# ソースコードをコピー
COPY src/. ./src/

# プロダクションビルドを実行
RUN NODE_ENV=production npm run build

# 本番環境用のステージ
FROM nginx:alpine

# 必要なパッケージをインストール
RUN apk add --no-cache curl

# nginxの実行に必要なディレクトリを作成し、適切な権限を設定
RUN mkdir -p /var/run/nginx /var/cache/nginx /var/log/nginx && \
    touch /var/run/nginx/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx /var/cache/nginx /var/log/nginx && \
    chmod -R 755 /var/run/nginx /var/cache/nginx /var/log/nginx

# Nginxの設定ファイルをコピー
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN chown -R nginx:nginx /etc/nginx/conf.d/default.conf && \
    chmod 644 /etc/nginx/conf.d/default.conf && \
    sed -i 's#pid.*#pid /var/run/nginx/nginx.pid;#g' /etc/nginx/nginx.conf

# ビルド成果物をNginxの公開ディレクトリにコピー
COPY --from=build /app/dist /usr/share/nginx/html

# セキュリティとパーミッションの設定
RUN chmod -R 755 /usr/share/nginx/html && \
    mkdir -p /usr/share/nginx/html/credential && \
    chmod 750 /usr/share/nginx/html/credential && \
    chown -R nginx:nginx /usr/share/nginx/html && \
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
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Nginxをフォアグラウンドで実行
CMD ["nginx", "-g", "daemon off;"]