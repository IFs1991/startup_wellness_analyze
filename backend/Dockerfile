# ====================================
# Builder ステージ: 依存関係のインストールと環境設定
# ====================================
FROM condaforge/mambaforge:latest as builder

# 環境変数の設定
ENV TZ=Asia/Tokyo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# システム依存関係のインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    git \
    cmake \
    pkg-config \
    wget \
    tzdata \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    libffi-dev \
    shared-mime-info \
    libfreetype6-dev \
    libpng-dev \
    libfontconfig1 \
    libjpeg-dev \
    libtiff5-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    locales \
    fonts-dejavu \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ロケールの設定
RUN locale-gen ja_JP.UTF-8
ENV LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8

# ワークディレクトリの設定
WORKDIR /app

# 環境ファイルのコピーと環境の作成
COPY environment.yml /app/
RUN mamba env create -f environment.yml && \
    mamba clean -afy

# Pip依存関係のコピーとインストール
COPY requirements.txt /app/requirements.txt
RUN mamba run -n startup_wellness_analyze pip install -r /app/requirements.txt && \
    mamba run -n startup_wellness_analyze pip install "email-validator>=2.0.0"

# アプリケーションファイルのコピー
COPY app/main.py /app/backend/main.py
COPY app/ /app/backend/app/
COPY core/ /app/backend/core/

# entrypoint.shを適切な場所にコピー
COPY app/entrypoint.sh /app/backend/app/entrypoint.sh
RUN chmod +x /app/backend/app/entrypoint.sh

# デバッグ情報の表示
RUN echo "=== Conda環境の確認 ===" && \
    mamba env list && \
    echo "=== Python バージョンの確認 ===" && \
    mamba run -n startup_wellness_analyze python --version && \
    echo "=== 依存パッケージの確認 ===" && \
    mamba run -n startup_wellness_analyze pip list && \
    echo "=== ディレクトリ構造の確認 ===" && \
    ls -la /app/backend/ && \
    echo "=== main.pyの場所の確認 ===" && \
    ls -la /app/backend/main.py

# ====================================
# ランタイムイメージ ステージ
# ====================================
FROM condaforge/miniforge3:latest

# 環境変数の設定
ENV TZ=Asia/Tokyo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    DEBIAN_FRONTEND=noninteractive

# システム依存関係のインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgcc-s1 \
    tzdata \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    shared-mime-info \
    libfreetype6 \
    libpng16-16 \
    libfontconfig1 \
    libjpeg-turbo8 \
    libtiff5 \
    libwebp6 \
    libopenjp2-7 \
    fonts-dejavu \
    curl \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ユーザー作成とワークディレクトリの設定
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app

# ビルドステージからファイルをコピー
COPY --from=builder /opt/conda/ /opt/conda/
COPY --from=builder /app/ /app/

# 必要なディレクトリ構造の作成
RUN mkdir -p /app/logs /app/data /app/credentials

# デバッグ情報の表示
RUN echo "=== 環境情報 ===" && \
    echo "Conda環境一覧:" && \
    conda env list && \
    echo "Pythonバージョン:" && \
    conda run -n startup_wellness_analyze python --version && \
    echo "ディレクトリ構造:" && \
    ls -la /app/backend/

# コンテナ起動設定
ENTRYPOINT ["/app/backend/app/entrypoint.sh"]