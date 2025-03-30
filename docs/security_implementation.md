# セキュリティ実装ガイド

## 必要な依存関係

セキュリティ機能を実装するために以下の依存関係が必要です。`requirements.txt`に追加してください：

```
# 認証関連
argon2-cffi==21.3.0       # パスワードハッシュ用
PyJWT==2.6.0              # JWT認証用
pyotp==2.8.0              # TOTP多要素認証用
qrcode==7.4.2             # MFA QRコード生成用
redis==4.5.4              # レート制限とセッション用
email-validator==2.0.0    # メール検証用
cryptography==39.0.1      # 暗号化機能用
httpx==0.24.0             # 非同期HTTPクライアント

# コンプライアンス関連
pydantic==1.10.7          # データバリデーション用
python-dateutil==2.8.2    # 日付処理用
```

## 設定ファイル

以下の内容で`backend/core/security_config.py`を作成してください：

```python
"""
セキュリティ設定モジュール
システム全体のセキュリティ関連設定を一元管理します
"""
import os
from typing import Dict, Any, List, Optional

# JWT設定
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-for-development-only")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7

# セッション設定
MAX_SESSIONS_PER_USER = 5
SESSION_TIMEOUT_MINUTES = 60
IDLE_TIMEOUT_MINUTES = 15

# パスワードポリシー
PASSWORD_MIN_LENGTH = 8
PASSWORD_REQUIRE_UPPERCASE = True
PASSWORD_REQUIRE_LOWERCASE = True
PASSWORD_REQUIRE_DIGIT = True
PASSWORD_REQUIRE_SPECIAL = True
PASSWORD_EXPIRY_DAYS = 90
PASSWORD_HISTORY_SIZE = 5

# レート制限設定
RATE_LIMIT_DEFAULT = "60/minute"
RATE_LIMIT_LOGIN = "5/minute"
RATE_LIMIT_REGISTER = "3/5minutes"
RATE_LIMIT_PASSWORD_RESET = "3/hour"
RATE_LIMIT_MFA = "5/5minutes"

# MFA設定
MFA_SETUP_EXPIRY_MINUTES = 10
MFA_TOKEN_EXPIRY_MINUTES = 5

# データ保持ポリシー
AUDIT_LOG_RETENTION_DAYS = 365
USER_DATA_RETENTION_DAYS = 730
INACTIVE_ACCOUNT_DAYS = 730

# コンプライアンス設定
COMPLIANCE_LOG_FORMAT = "json"
COMPLIANCE_LOG_PATH = "logs/compliance"

def get_secret_key() -> str:
    """アプリケーションのシークレットキーを取得します"""
    return os.getenv("APP_SECRET_KEY", JWT_SECRET_KEY)
```

## 未定義の関数と変数の修正

### `get_secret_key`関数

この関数は上記の`security_config.py`ファイルで実装します。
次のインポートを`auth_manager.py`に追加してください：

```python
from .security_config import get_secret_key
```

### `user`変数の未定義エラー

これらは関数パラメータまたはローカルスコープで定義される変数の使用ミスです。
以下の行を修正してください：

1. 535行目:
```python
def update_user_role(self, user_id: str, new_role: str) -> bool:
    """ユーザーのロールを更新"""
    try:
        # userをuser_idに修正
        user_data = self.db.get_user_by_id(user_id)
        if not user_data:
            return False
```

2. 1130行目:
```python
def _verify_user_credentials(self, user_id: str, password: str) -> Optional[Dict[str, Any]]:
    """ユーザーの認証情報を検証"""
    # userをuser_idに修正
    user_data = self.db.get_user_by_id(user_id)
    if not user_data:
        return None
```

## 実装手順

1. `requirements.txt`に上記の依存関係を追加し、インストールします：
```bash
pip install -r requirements.txt
```

2. `security_config.py`を作成します。

3. `auth_manager.py`の未定義変数のエラーを修正します。

4. テスト実行して、エラーが解消されたことを確認します。

## セキュリティ設定の変更方法

本番環境では必ず以下の環境変数を設定してください：

```bash
# 本番環境設定例
export JWT_SECRET_KEY="extremely-secure-random-key-here"
export APP_SECRET_KEY="another-secure-key-for-application"
```

## その他の注意点

1. セキュリティに関連するキーやシークレットは環境変数として設定し、コードにハードコーディングしないでください。

2. 本番環境では必ずHTTPSを使用してください。

3. `.env`ファイルを使用する場合は、バージョン管理に含めないでください（`.gitignore`に追加）。