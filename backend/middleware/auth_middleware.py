"""
認証ミドルウェア
ユーザーの認証を処理し、リクエストに認証情報を添付します。
"""
from fastapi import Request, HTTPException, Depends, status
from firebase_admin import auth, exceptions
import logging
from typing import Dict, Any, Optional
from src.database.firestore.client import FirestoreClient

logger = logging.getLogger(__name__)
firestore_client = FirestoreClient()

async def verify_token(token: str) -> Dict[str, Any]:
    """
    Firebaseトークンを検証し、デコードされたトークン情報を返す

    Args:
        token: Firebase認証トークン

    Returns:
        デコードされたトークン情報

    Raises:
        HTTPException: トークンが無効な場合
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証トークンがありません"
        )

    try:
        # Firebaseトークンを検証
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except exceptions.FirebaseError as e:
        logger.error(f"Firebase認証エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無効な認証トークンです"
        )
    except Exception as e:
        logger.error(f"認証エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証処理中にエラーが発生しました"
        )

async def get_current_user(request: Request) -> Dict[str, Any]:
    """
    現在のリクエストからユーザー情報を取得する

    Args:
        request: FastAPIリクエスト

    Returns:
        現在のユーザー情報

    Raises:
        HTTPException: ユーザーが認証されていない場合
    """
    # リクエストステートからユーザーを取得（auth_middlewareで設定）
    user = getattr(request.state, 'user', None)

    if not user:
        # 認証ヘッダーから手動で取得を試みる
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if token:
            user = await verify_token(token)
            # 取得したユーザー情報をリクエストステートに設定
            request.state.user = user
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="認証されていません"
            )

    return user

async def get_user_data(user_id: str) -> Optional[Dict[str, Any]]:
    """
    ユーザーIDからFirestoreに保存されたユーザーデータを取得

    Args:
        user_id: ユーザーID

    Returns:
        ユーザーデータ（存在する場合）
    """
    try:
        user_data = await firestore_client.get_document(
            collection='users',
            doc_id=user_id
        )
        return user_data
    except Exception as e:
        logger.error(f"ユーザーデータ取得エラー: {str(e)}")
        return None

async def get_complete_user_info(user_id: str) -> Dict[str, Any]:
    """
    ユーザーの完全な情報（認証情報とデータベース情報）を取得

    Args:
        user_id: ユーザーID

    Returns:
        ユーザーの完全な情報
    """
    try:
        # Firebaseの認証情報を取得
        auth_info = auth.get_user(user_id)

        # Firestoreのユーザーデータを取得
        user_data = await get_user_data(user_id)

        if not user_data:
            return {
                "uid": auth_info.uid,
                "email": auth_info.email,
                "display_name": auth_info.display_name,
                "photo_url": auth_info.photo_url,
                "email_verified": auth_info.email_verified,
                "disabled": auth_info.disabled,
                "db_data": None
            }

        # 認証情報とユーザーデータを統合
        return {
            "uid": auth_info.uid,
            "email": auth_info.email,
            "display_name": auth_info.display_name,
            "photo_url": auth_info.photo_url,
            "email_verified": auth_info.email_verified,
            "disabled": auth_info.disabled,
            "db_data": user_data
        }
    except exceptions.FirebaseError as e:
        logger.error(f"ユーザー情報取得エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ユーザーが見つかりません"
        )
    except Exception as e:
        logger.error(f"ユーザー情報取得中のエラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ユーザー情報の取得中にエラーが発生しました"
        )

def admin_required(request: Request):
    """
    管理者権限が必要なエンドポイント用のデコレータ

    Args:
        request: FastAPIリクエスト

    Returns:
        管理者かどうか

    Raises:
        HTTPException: ユーザーが管理者でない場合
    """
    user = request.state.user

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証されていません"
        )

    # 管理者権限の確認（カスタムクレームまたはDBデータを使用）
    is_admin = user.get("admin", False)

    if not is_admin:
        # Firestoreからユーザーデータを取得して確認
        user_data = get_user_data(user.get("uid"))
        is_admin = user_data and user_data.get("is_admin", False)

    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="管理者権限が必要です"
        )

    return True