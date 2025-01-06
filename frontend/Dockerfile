# ビルドステージ
FROM node:18-alpine as build

WORKDIR /app

# 依存関係のインストール
COPY package*.json ./
RUN npm install

# ソースコードのコピーとビルド
COPY . .
RUN npm run build

# 本番ステージ
FROM nginx:alpine

# ビルドしたファイルをNginxのドキュメントルートにコピー
COPY --from=build /app/dist /usr/share/nginx/html

# Nginxの設定ファイルをコピー
COPY nginx.conf /etc/nginx/conf.d/default.conf

# ポートの設定
EXPOSE 80
ENV PORT=80

CMD ["nginx", "-g", "daemon off;"]