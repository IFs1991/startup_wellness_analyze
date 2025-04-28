# ビルドステージ - 依存関係インストール用
FROM node:18-alpine AS deps

WORKDIR /app

# 依存関係ファイルのみをコピー
COPY package.json pnpm-lock.yaml* ./

# 必要なビルドツールを最小限に抑える
RUN apk add --no-cache --virtual .build-deps g++ && \
    npm install -g pnpm && \
    pnpm install --no-frozen-lockfile && \
    pnpm store prune && \
    apk del .build-deps

# ビルドステージ - アプリケーションのビルド
FROM node:18-alpine AS builder

# pnpmをグローバルインストール
RUN npm install -g pnpm

# ビルド時の環境変数設定
ARG NODE_ENV=production
ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_FIREBASE_CONFIG

ENV NODE_ENV=${NODE_ENV} \
    NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    NEXT_PUBLIC_FIREBASE_CONFIG=${NEXT_PUBLIC_FIREBASE_CONFIG} \
    PATH=/app/node_modules/.bin:$PATH

WORKDIR /app

# 設定ファイルとpackage.jsonをコピー
COPY package.json pnpm-lock.yaml* ./
COPY tsconfig*.json ./
COPY next.config.mjs ./
COPY .env* ./

# 依存関係をdepsステージからコピー
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /root/.local/share/pnpm/store ./pnpm-store

# ソースコードをコピー
COPY . .

# TypeScriptのコンパイルとビルド
RUN ls -la && \
    pnpm build && \
    find /app/.next -type f -name '*.map' -delete

# 本番環境用の最小限のイメージ
FROM node:18-alpine AS runner

# セキュリティとパフォーマンスのための環境変数を設定
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

WORKDIR /app

# 必要なファイルのみをコピー
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# エントリーポイントスクリプト
COPY scripts/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# ヘルスチェックスクリプト
COPY healthcheck.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/healthcheck.sh

# コンテナ内のポート3000を公開
EXPOSE 3000

# ヘルスチェックを直接定義
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -s -f http://localhost:3000/ > /dev/null || exit 1

# nodeユーザーで実行
USER node

# エントリーポイントを設定してNext.jsを起動
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["node", "server.js"]