FROM nginx:alpine

# セキュリティ設定
RUN apk add --no-cache curl && \
    rm -rf /var/cache/apk/* && \
    rm -rf /etc/nginx/conf.d/*

# 公開するファイルの作成
RUN mkdir -p /usr/share/nginx/html
RUN echo "<html><head><title>Startup Wellness Analyzer</title></head><body><h1>Startup Wellness Analyzer</h1><p>PostgreSQL and Redis services are running.</p></body></html>" > /usr/share/nginx/html/index.html

# Nginxの設定ファイル
COPY nginx.conf /etc/nginx/conf.d/default.conf

# コンテナ内のポート80を公開
EXPOSE 80

# Nginxを起動
CMD ["nginx", "-g", "daemon off;"]