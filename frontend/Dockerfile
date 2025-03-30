# ビルドステージ - 依存関係インストール用
FROM node:18-alpine AS deps

WORKDIR /app

# 依存関係ファイルのみをコピー
COPY package.json yarn.lock* ./

# 必要なビルドツールを最小限に抑える
RUN apk add --no-cache --virtual .build-deps g++ && \
    yarn install --frozen-lockfile --network-timeout 600000 && \
    yarn cache clean && \
    apk del .build-deps

# ビルドステージ - アプリケーションのビルド
FROM node:18-alpine AS builder

# ビルド時の環境変数設定
ARG NODE_ENV=production
ARG VITE_API_URL
ARG VITE_FIREBASE_CONFIG

ENV NODE_ENV=${NODE_ENV} \
    VITE_API_URL=${VITE_API_URL} \
    VITE_FIREBASE_CONFIG=${VITE_FIREBASE_CONFIG} \
    PATH=/app/node_modules/.bin:$PATH

WORKDIR /app

# 設定ファイルとpackage.jsonをコピー
COPY package.json yarn.lock* ./
COPY tsconfig*.json ./
COPY vite.config.ts ./
COPY index.html ./
COPY .env* ./

# 依存関係をdepsステージからコピー
COPY --from=deps /app/node_modules ./node_modules

# ソースコードをコピー
COPY src/ ./src/

# TypeScriptのコンパイルとビルド
RUN ls -la && \
    yarn build && \
    find /app/dist -type f -name '*.map' -delete

# 本番環境用の最小限のイメージ
FROM nginx:stable-alpine AS runner

# セキュリティとパフォーマンスのための環境変数を設定
ENV NGINX_WORKER_PROCESSES=auto \
    NGINX_WORKER_CONNECTIONS=1024 \
    NGINX_WORKER_RLIMIT_NOFILE=4096 \
    MALLOC_TRIM_THRESHOLD_=65536

# 必要なディレクトリ構造の一括作成（重複を排除）
RUN apk add --no-cache curl && \
    rm -rf /etc/nginx/conf.d/* && \
    mkdir -p /var/cache/nginx/client_temp \
             /var/cache/nginx/proxy_temp \
             /var/cache/nginx/fastcgi_temp \
             /var/cache/nginx/uwsgi_temp \
             /var/cache/nginx/scgi_temp \
             /var/run/nginx \
             /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chmod 755 /var/run/nginx.pid && \
    chown -R nginx:nginx /var/cache/nginx \
                         /var/run/nginx \
                         /var/log/nginx \
                         /var/run/nginx.pid && \
    # キャッシュとパッケージリストを削除してイメージサイズを縮小
    rm -rf /var/cache/apk/*

# ビルド済みファイルをNginxのドキュメントルートにコピー
COPY --from=builder /app/dist /usr/share/nginx/html
RUN chown -R nginx:nginx /usr/share/nginx/html

# Nginxの設定ファイル
COPY nginx.conf /etc/nginx/conf.d/default.conf

# エントリーポイントスクリプト（最小限にした）
COPY scripts/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# コンテナ内のポート80を公開
EXPOSE 80

# ヘルスチェックを直接定義（スクリプトファイルなし）
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -s -f http://localhost/ > /dev/null || exit 1

# nginxユーザーで実行
USER nginx

# エントリーポイントを設定してnginxを起動
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []