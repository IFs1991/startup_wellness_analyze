#!/bin/bash
# スタートアップ分析プラットフォーム: mamba/conda環境セットアップスクリプト
# 高速な依存関係解決と効率的な環境管理を実現します

# エラーハンドリング
set -e

# 現在のディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# カラー出力設定
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ヘルパー関数
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    }
}

print_header "スタートアップ分析プラットフォーム: 環境セットアップを開始します"

# OS検出
OS_TYPE=$(detect_os)
print_success "検出されたOS: $OS_TYPE"

# miniforgeのインストール確認
install_miniforge() {
    print_header "Miniforgeのインストール"

    if [[ "$OS_TYPE" == "linux" ]]; then
        # Linux
        curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        bash Miniforge3-$(uname)-$(uname -m).sh -b
        rm Miniforge3-$(uname)-$(uname -m).sh
        ~/miniforge3/bin/conda init bash
        export PATH="$HOME/miniforge3/bin:$PATH"
        print_success "Miniforgeがインストールされました"
        echo "新しいターミナルを開くか、'source ~/.bashrc'を実行してPATHを更新してください"
    elif [[ "$OS_TYPE" == "macos" ]]; then
        # macOS
        curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$(uname -m).sh"
        bash Miniforge3-MacOSX-$(uname -m).sh -b
        rm Miniforge3-MacOSX-$(uname -m).sh
        ~/miniforge3/bin/conda init "$(basename "${SHELL}")"
        export PATH="$HOME/miniforge3/bin:$PATH"
        print_success "Miniforgeがインストールされました"
        echo "新しいターミナルを開くか、'source ~/.$(basename "${SHELL}")rc'を実行してPATHを更新してください"
    elif [[ "$OS_TYPE" == "windows" ]]; then
        # Windows
        print_warning "Windowsでは手動でMiniforgeをインストールしてください: https://github.com/conda-forge/miniforge"
        print_warning "インストール後、このスクリプトを再実行してください"
        exit 0
    else
        print_error "サポートされていないOSタイプ: $OS_TYPE"
        exit 1
    fi
}

if ! command -v conda &> /dev/null; then
    print_warning "Condaが見つかりません。Miniforgeをインストールします..."
    install_miniforge

    # インストール後に環境変数を更新
    if [[ "$OS_TYPE" == "linux" || "$OS_TYPE" == "macos" ]]; then
        # インストールスクリプトを実行後はcondaコマンドが使えないかもしれないので、絶対パスを使用
        if [[ -f "$HOME/miniforge3/bin/conda" ]]; then
            CONDA_EXE="$HOME/miniforge3/bin/conda"
        else
            print_error "Condaのインストールパスが見つかりません"
            exit 1
        fi
    else
        print_error "Condaがインストールされていないようです"
        exit 1
    fi
else
    CONDA_EXE=$(which conda)
    print_success "既存のCondaが見つかりました: $CONDA_EXE"
fi

# mambaのインストール
install_mamba() {
    print_header "mambaのインストール"
    $CONDA_EXE install -y -n base -c conda-forge mamba
    print_success "mambaがインストールされました"
}

if ! command -v mamba &> /dev/null; then
    print_warning "mambaが見つかりません。インストールします..."
    install_mamba

    # インストール後にmambaコマンドのパスを確認
    if [[ -f "$HOME/miniforge3/bin/mamba" ]]; then
        MAMBA_EXE="$HOME/miniforge3/bin/mamba"
    else
        # condaの場所からmambaの場所を推測
        CONDA_DIR=$(dirname "$CONDA_EXE")
        if [[ -f "$CONDA_DIR/mamba" ]]; then
            MAMBA_EXE="$CONDA_DIR/mamba"
        else
            print_warning "mambaのインストールパスが見つかりません。condaを使用します"
            MAMBA_EXE="$CONDA_EXE"
        fi
    fi
else
    MAMBA_EXE=$(which mamba)
    print_success "既存のmambaが見つかりました: $MAMBA_EXE"
fi

# 既存の環境を確認
print_header "環境の確認"
if $CONDA_EXE env list | grep -q "causal-analytics"; then
    print_warning "causal-analytics環境は既に存在します"

    read -p "既存の環境を更新しますか？ (y) 更新 / (r) 再作成 / (s) スキップ: " ACTION
    case $ACTION in
        [Yy]*)
            print_header "既存の環境を更新します..."
            $MAMBA_EXE env update -f environment.yml
            print_success "環境を更新しました"
            ;;
        [Rr]*)
            print_header "既存の環境を削除します..."
            $CONDA_EXE deactivate
            $CONDA_EXE env remove -n causal-analytics
            print_success "既存の環境を削除しました"

            print_header "mambaを使用して高速に環境を作成します..."
            $MAMBA_EXE env create -f environment.yml
            print_success "環境を作成しました"
            ;;
        *)
            print_warning "環境の更新をスキップします"
            ;;
    esac
else
    print_header "新規環境を作成します"
    print_warning "環境の作成には数分かかる場合があります"
    $MAMBA_EXE env create -f environment.yml
    print_success "環境を作成しました"
fi

# 東京リージョンのスケジュール設定
print_header "東京リージョン (asia-northeast1) の割引時間帯設定"
echo "割引時間帯:"
echo "  平日: 22:00-08:00 (JST)"
echo "  週末: 終日"
print_success "これらの時間帯に重い処理をスケジュールすることでコストを削減できます"

# GCPプロジェクト設定の確認（オプション）
print_header "GCP設定の確認（オプション）"
read -p "GCPプロジェクトIDを設定しますか？ (y/n): " SET_GCP
if [[ $SET_GCP == [Yy]* ]]; then
    read -p "GCPプロジェクトIDを入力してください: " GCP_PROJECT_ID

    # .envファイルを作成または更新
    ENV_FILE=".env"
    if [[ -f "$ENV_FILE" ]]; then
        # GCP_PROJECT_IDエントリがあれば更新、なければ追加
        if grep -q "GCP_PROJECT_ID=" "$ENV_FILE"; then
            sed -i.bak "s/GCP_PROJECT_ID=.*/GCP_PROJECT_ID=$GCP_PROJECT_ID/" "$ENV_FILE"
        else
            echo "GCP_PROJECT_ID=$GCP_PROJECT_ID" >> "$ENV_FILE"
        fi
    else
        # ファイルが存在しない場合は新規作成
        echo "GCP_PROJECT_ID=$GCP_PROJECT_ID" > "$ENV_FILE"
    fi

    print_success "GCPプロジェクトID ($GCP_PROJECT_ID) を.envファイルに設定しました"
fi

# 環境診断
run_diagnostics() {
    print_header "環境診断を実行します"

    # 環境を有効化して診断を実行
    $CONDA_EXE activate causal-analytics

    echo "Python バージョン:"
    python --version

    echo -e "\n主要パッケージのバージョン:"
    python -c "import pandas; print(f'pandas: {pandas.__version__}')"
    python -c "import numpy; print(f'numpy: {numpy.__version__}')"
    python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"

    echo -e "\nGPUサポート状況:"
    python -c "
try:
    import tensorflow as tf
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    print(f'TensorFlow GPUサポート: {\"有効\" if gpu_available else \"無効\"}')
except:
    print('TensorFlow GPUサポート: インストールエラーまたは未インストール')

try:
    import torch
    print(f'PyTorch GPUサポート: {\"有効\" if torch.cuda.is_available() else \"無効\"}')
except:
    print('PyTorch GPUサポート: インストールエラーまたは未インストール')
"

    echo -e "\n環境情報:"
    python -c "
import sys
import platform
print(f'Pythonパス: {sys.executable}')
print(f'プラットフォーム: {platform.platform()}')
print(f'プロセッサ: {platform.processor()}')
"

    $CONDA_EXE deactivate
}

read -p "環境診断を実行しますか？ (y/n): " RUN_DIAG
if [[ $RUN_DIAG == [Yy]* ]]; then
    run_diagnostics
fi

# 環境構築完了
print_header "環境構築が完了しました"
echo "次のコマンドで環境を有効化できます: conda activate causal-analytics"
echo ""
echo "コスト最適化のためのヒント:"
echo "1. 東京リージョン (asia-northeast1) の割引時間帯 (平日22:00-08:00, 週末終日) に重い処理をスケジュールする"
echo "2. backend/utils/dask_optimizer.py を使用して大規模データ処理を最適化する"
echo "3. backend/federated_learning/scheduler/optimal_scheduling.py で連合学習を最適にスケジュールする"
echo ""
echo "詳細は docs/mamba_conda_guide.md を参照してください"

# セットアップをWindowsでも使いやすくするためのPowerShellスクリプトを生成
if [[ "$OS_TYPE" == "windows" ]]; then
    print_header "Windowsユーザー向けのPowerShellスクリプトを生成"

    # PowerShellスクリプトの生成
    cat > setup_environment.ps1 << 'EOF'
# スタートアップ分析プラットフォーム: mamba/conda環境セットアップスクリプト (Windows PowerShell版)

# 関数定義
function Write-Header($text) {
    Write-Host "`n=== $text ===`n" -ForegroundColor Blue
}

function Write-Success($text) {
    Write-Host "✓ $text" -ForegroundColor Green
}

function Write-Warning($text) {
    Write-Host "⚠ $text" -ForegroundColor Yellow
}

function Write-Error($text) {
    Write-Host "✗ $text" -ForegroundColor Red
}

# メイン処理
Write-Header "スタートアップ分析プラットフォーム: 環境セットアップを開始します"

# condaコマンドの確認
$condaExists = $null -ne (Get-Command conda -ErrorAction SilentlyContinue)
if (-not $condaExists) {
    Write-Warning "Condaが見つかりません。Miniforgeをインストールしてください: https://github.com/conda-forge/miniforge"
    Write-Warning "インストール後、このスクリプトを再実行してください"
    exit
} else {
    $condaPath = (Get-Command conda).Source
    Write-Success "既存のCondaが見つかりました: $condaPath"
}

# mambaコマンドの確認
$mambaExists = $null -ne (Get-Command mamba -ErrorAction SilentlyContinue)
if (-not $mambaExists) {
    Write-Warning "mambaが見つかりません。インストールします..."
    conda install -y -n base -c conda-forge mamba
    Write-Success "mambaがインストールされました"
    $mambaPath = (Join-Path (Split-Path $condaPath) "mamba.exe")
} else {
    $mambaPath = (Get-Command mamba).Source
    Write-Success "既存のmambaが見つかりました: $mambaPath"
}

# 既存の環境を確認
Write-Header "環境の確認"
$envExists = conda env list | Select-String "causal-analytics"
if ($envExists) {
    Write-Warning "causal-analytics環境は既に存在します"

    $action = Read-Host "既存の環境を更新しますか？ (y) 更新 / (r) 再作成 / (s) スキップ"
    switch -Regex ($action) {
        "[Yy].*" {
            Write-Header "既存の環境を更新します..."
            & mamba env update -f environment.yml
            Write-Success "環境を更新しました"
        }
        "[Rr].*" {
            Write-Header "既存の環境を削除します..."
            conda deactivate
            conda env remove -n causal-analytics
            Write-Success "既存の環境を削除しました"

            Write-Header "mambaを使用して高速に環境を作成します..."
            & mamba env create -f environment.yml
            Write-Success "環境を作成しました"
        }
        default {
            Write-Warning "環境の更新をスキップします"
        }
    }
} else {
    Write-Header "新規環境を作成します"
    Write-Warning "環境の作成には数分かかる場合があります"
    & mamba env create -f environment.yml
    Write-Success "環境を作成しました"
}

# 東京リージョンのスケジュール設定
Write-Header "東京リージョン (asia-northeast1) の割引時間帯設定"
Write-Host "割引時間帯:"
Write-Host "  平日: 22:00-08:00 (JST)"
Write-Host "  週末: 終日"
Write-Success "これらの時間帯に重い処理をスケジュールすることでコストを削減できます"

# GCPプロジェクト設定の確認（オプション）
Write-Header "GCP設定の確認（オプション）"
$setGcp = Read-Host "GCPプロジェクトIDを設定しますか？ (y/n)"
if ($setGcp -match "[Yy].*") {
    $gcpProjectId = Read-Host "GCPプロジェクトIDを入力してください"

    # .envファイルを作成または更新
    $envFile = ".env"
    if (Test-Path $envFile) {
        $envContent = Get-Content $envFile
        $updated = $false
        $newContent = @()

        foreach ($line in $envContent) {
            if ($line -match "GCP_PROJECT_ID=") {
                $newContent += "GCP_PROJECT_ID=$gcpProjectId"
                $updated = $true
            } else {
                $newContent += $line
            }
        }

        if (-not $updated) {
            $newContent += "GCP_PROJECT_ID=$gcpProjectId"
        }

        $newContent | Set-Content $envFile
    } else {
        "GCP_PROJECT_ID=$gcpProjectId" | Set-Content $envFile
    }

    Write-Success "GCPプロジェクトID ($gcpProjectId) を.envファイルに設定しました"
}

# 環境診断
function Run-Diagnostics {
    Write-Header "環境診断を実行します"

    # 環境を有効化して診断を実行
    & conda activate causal-analytics

    Write-Host "Python バージョン:"
    python --version

    Write-Host "`n主要パッケージのバージョン:"
    python -c "import pandas; print(f'pandas: {pandas.__version__}')"
    python -c "import numpy; print(f'numpy: {numpy.__version__}')"
    python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"

    Write-Host "`nGPUサポート状況:"
    python -c "
try:
    import tensorflow as tf
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    print(f'TensorFlow GPUサポート: {\"有効\" if gpu_available else \"無効\"}')
except:
    print('TensorFlow GPUサポート: インストールエラーまたは未インストール')

try:
    import torch
    print(f'PyTorch GPUサポート: {\"有効\" if torch.cuda.is_available() else \"無効\"}')
except:
    print('PyTorch GPUサポート: インストールエラーまたは未インストール')
"

    Write-Host "`n環境情報:"
    python -c "
import sys
import platform
print(f'Pythonパス: {sys.executable}')
print(f'プラットフォーム: {platform.platform()}')
print(f'プロセッサ: {platform.processor()}')
"

    & conda deactivate
}

$runDiag = Read-Host "環境診断を実行しますか？ (y/n)"
if ($runDiag -match "[Yy].*") {
    Run-Diagnostics
}

# 環境構築完了
Write-Header "環境構築が完了しました"
Write-Host "次のコマンドで環境を有効化できます: conda activate causal-analytics"
Write-Host ""
Write-Host "コスト最適化のためのヒント:"
Write-Host "1. 東京リージョン (asia-northeast1) の割引時間帯 (平日22:00-08:00, 週末終日) に重い処理をスケジュールする"
Write-Host "2. backend/utils/dask_optimizer.py を使用して大規模データ処理を最適化する"
Write-Host "3. backend/federated_learning/scheduler/optimal_scheduling.py で連合学習を最適にスケジュールする"
Write-Host ""
Write-Host "詳細は docs/mamba_conda_guide.md を参照してください"
EOF

    print_success "PowerShellスクリプト (setup_environment.ps1) を生成しました"
    echo "Windowsでは 'setup_environment.ps1' を右クリックして「PowerShellで実行」を選択してください"
fi