from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth
from typing import Dict, Any
from datetime import datetime
from src.database.firestore.client import FirestoreClient

router = APIRouter()
security = HTTPBearer()
firestore_client = FirestoreClient()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Firebaseトークンを検証し、ユーザー情報を取得

    Args:
        credentials: HTTPAuthorizationCredentials

    Returns:
        Dict[str, Any]: ユーザー情報

    Raises:
        HTTPException: 認証エラー
    """
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/register")
async def register_user(user_data: Dict[str, Any]):
    """
    新規ユーザー登録

    Args:
        user_data (Dict[str, Any]): ユーザー情報

    Returns:
        Dict[str, Any]: 登録結果
    """
    try:
        # Firebaseユーザーを作成
        user = auth.create_user(
            email=user_data.get("email"),
            password=user_data.get("password"),
            display_name=user_data.get("display_name"),
        )

        # Firestoreにユーザー情報を保存
        user_doc_data = {
            "email": user_data.get("email"),
            "display_name": user_data.get("display_name"),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        await firestore_client.create_document("users", user.uid, user_doc_data)

        return {"message": "ユーザーが正常に登録されました", "user_id": user.uid}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login")
async def login(credentials: Dict[str, Any]):
    """ログイン処理"""
    try:
        user_id = credentials.get('user_id')
        password = credentials.get('password')

        if not user_id or not password:
            raise HTTPException(status_code=400, detail="ユーザーIDとパスワードは必須です")

        # ユーザー認証
        user = await firestore_client.get_document(
            collection='users',
            doc_id=user_id
        )

        if not user or user.get('password') != password:  # 本番環境ではハッシュ化が必要
            raise HTTPException(status_code=401, detail="認証に失敗しました")

        return {
            "status": "success",
            "user": {
                "id": user_id,
                "name": user.get('name'),
                "role": user.get('role')
            }
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/logout")
async def logout():
    """ログアウト処理"""
    return {"status": "success"}

@router.get("/me")
async def get_current_user(token_data: Dict[str, Any] = Depends(verify_token)):
    """
    現在のユーザー情報を取得

    Args:
        token_data (Dict[str, Any]): トークンデータ

    Returns:
        Dict[str, Any]: ユーザー情報
    """
    try:
        user_id = token_data.get("uid")
        user_data = await firestore_client.get_document("users", user_id)

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="ユーザーが見つかりません"
            )

        return user_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/me")
async def update_user(
    user_data: Dict[str, Any],
    token_data: Dict[str, Any] = Depends(verify_token)
):
    """
    ユーザー情報を更新

    Args:
        user_data (Dict[str, Any]): 更新するユーザー情報
        token_data (Dict[str, Any]): トークンデータ

    Returns:
        Dict[str, str]: 更新結果
    """
    try:
        user_id = token_data.get("uid")
        await firestore_client.update_document("users", user_id, user_data)
        return {"message": "ユーザー情報が更新されました"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )