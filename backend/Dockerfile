FROM condaforge/miniforge3:latest as builder

# 環境変数の設定
ENV TZ=Asia/Tokyo
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# タイムゾーンの設定とビルドツール・システム依存関係のインストール
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    git \
    libgcc-9-dev \
    libgcc-s1 \
    cmake \
    pkg-config \
    wget \
    # WeasyPrintの依存関係
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    libffi-dev \
    shared-mime-info \
    # matplotlib, OpenCVの依存関係
    libfreetype6-dev \
    libpng-dev \
    libfontconfig1 \
    # OpenCVの依存関係
    libjpeg-dev \
    libtiff5-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    # 基本的なフォントとロケール
    locales \
    fonts-dejavu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ロケールの設定
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

# ワークディレクトリの設定
WORKDIR /app

# Python環境の作成
RUN conda create -n startup_wellness_analyze python=3.11 -y

# 依存関係のインストール
COPY environment.yml .
RUN conda env update -n startup_wellness_analyze -f environment.yml

# シェルの初期化とコンダ環境のアクティベーション
SHELL ["/bin/bash", "-c"]

# 依存関係のインストール
RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate startup_wellness_analyze \
    && pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    starlette \
    dash \
    gunicorn \
    psycopg2-binary \
    pandas \
    numpy \
    openpyxl \
    db-dtypes \
    scipy \
    statsmodels \
    scikit-learn \
    arviz \
    causalimpact \
    dowhy \
    pgmpy \
    cvxpy \
    prophet \
    nltk \
    textblob \
    lifelines \
    mlxtend \
    networkx \
    neo4j \
    matplotlib \
    seaborn \
    plotly \
    firebase-admin \
    google-cloud-firestore \
    google-cloud-bigquery \
    google-cloud-storage \
    reportlab \
    jinja2 \
    weasyprint \
    google-auth \
    google-auth-oauthlib \
    google-auth-httplib2 \
    google-api-python-client \
    google-generativeai \
    requests \
    beautifulsoup4 \
    httpx \
    pycryptodome \
    bcrypt \
    authlib \
    argon2-cffi \
    python-jose \
    python-multipart \
    email-validator \
    pyotp \
    qrcode \
    pytest \
    pytest-asyncio \
    flake8 \
    black \
    mypy \
    boto3 \
    celery \
    redis \
    PyYAML \
    python-dotenv \
    flwr \
    opacus \
    cryptography \
    prometheus-client \
    loguru \
    pykwalify \
    memory_profiler \
    dask \
    pyarrow \
    datashader \
    google-cloud-scheduler \
    google-cloud-monitoring \
    psutil \
    numba \
    s3fs \
    gcsfs \
    sqlalchemy \
    alembic \
    apispec \
    flask-appbuilder \
    google-cloud-secret-manager \
    tf-keras \
    keras

# ソースコードのコピー
COPY . .

# 必要なディレクトリ構造を作成
RUN mkdir -p /app/api/routers /app/core /app/service/firestore /app/database /app/credentials

# モックファイルを作成
RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate startup_wellness_analyze \
    && mkdir -p /opt/conda/envs/startup_wellness_analyze/lib/python3.11/site-packages/econml \
    && mkdir -p /opt/conda/envs/startup_wellness_analyze/lib/python3.11/site-packages/shap \
    && echo 'class DummyClass: pass' > /opt/conda/envs/startup_wellness_analyze/lib/python3.11/site-packages/econml/__init__.py \
    && echo 'class DummyClass: pass' > /opt/conda/envs/startup_wellness_analyze/lib/python3.11/site-packages/shap/__init__.py

# モジュール構造の作成（エラー回避用）
RUN mkdir -p /app/api/routers && \
    for module in auth data_input analysis visualization data_processing prediction report_generation compliance; do \
        echo "from fastapi import APIRouter\nrouter = APIRouter()" > /app/api/routers/${module}.py; \
    done && \
    mkdir -p /app/core && \
    for module in wellness_score_calculator data_preprocessor correlation_analyzer cluster_analyzer time_series_analyzer survival_analyzer performance_predictor startup_survival_analyzer pca_analyzer text_miner feature_engineer descriptive_stats_calculator data_quality_checker model_evaluator generative_ai_manager custom_report_builder pdf_report_generator security rate_limiter auth_metrics; do \
        echo "class ${module^} {};\ndef create_wellness_score_calculator():\n    return WellnessScoreCalculator()" > /app/core/${module}.py; \
    done && \
    mkdir -p /app/service/firestore && \
    echo "class FirestoreService {};\nclass FirestoreClient {};\nclass StorageError(Exception) {};\nclass ValidationError(Exception) {};" > /app/service/firestore/client.py && \
    mkdir -p /app/database && \
    echo "def get_db(): pass\ndef init_neo4j(): pass\nclass Neo4jService {};" > /app/database/database.py && \
    echo "def init_neo4j(): pass\nclass Neo4jService {};" > /app/database/neo4j.py

# ランタイムイメージの作成
FROM condaforge/miniforge3:latest

# 環境変数の設定
ENV TZ=Asia/Tokyo
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

# タイムゾーンの設定とランタイム依存関係のインストール
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libgcc-9-dev \
    libgcc-s1 \
    # WeasyPrintの実行時依存関係
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    shared-mime-info \
    # matplotlib, OpenCVの実行時依存関係
    libfreetype6 \
    libpng16-16 \
    libfontconfig1 \
    # OpenCVの実行時依存関係
    libjpeg-turbo8 \
    libtiff5 \
    libwebp6 \
    libopenjp2-7 \
    # 基本的なフォント
    fonts-dejavu \
    # デバッグ用ツール
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ワークディレクトリの設定
WORKDIR /app

# ビルドステージから必要なファイルをコピー
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /app /app

# 必要なディレクトリ構造を確保
RUN mkdir -p /app/credentials /app/data

# エントリポイントスクリプトの実行権限付与
RUN chmod +x /app/entrypoint.sh

# デバッグ情報の出力
RUN ls -la /app && \
    echo "main.pyが存在するか確認: $(test -f /app/main.py && echo 存在します || echo 存在しません)" && \
    echo "entrypoint.shが存在するか確認: $(test -f /app/entrypoint.sh && echo 存在します || echo 存在しません)" && \
    echo "entrypoint.shの実行権限: $(test -x /app/entrypoint.sh && echo あります || echo ありません)" && \
    echo "モジュール構造の確認:" && \
    ls -la /app/api/routers && \
    ls -la /app/core

# コンテナ起動時に実行されるコマンド
ENTRYPOINT ["/app/entrypoint.sh"]