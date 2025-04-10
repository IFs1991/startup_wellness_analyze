"""
Redisユーザーリポジトリ

ユーザー情報をRedisにキャッシュするリポジトリの実装。
"""

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.domain.models.user import User, UserProfile, UserCredentials, UserRole
from backend.domain.repositories.user_repository import UserRepositoryInterface
from backend.core.exceptions import UserNotFoundError
from backend.infrastructure.redis.redis_service import RedisService

logger = logging.getLogger(__name__)


class RedisUserRepository(UserRepositoryInterface):
    """
    Redisを使用したユーザーデータのキャッシュリポジトリ実装。
    デコレータパターンを使用して、メインリポジトリの前にキャッシュ層として機能します。
    """

    # キャッシュキーのプレフィックス
    USER_KEY_PREFIX = "user:"
    EMAIL_INDEX_PREFIX = "user:email:"
    USERNAME_INDEX_PREFIX = "user:username:"

    def __init__(
        self,
        redis_service: RedisService,
        main_repository: UserRepositoryInterface,
        ttl_seconds: int = 3600  # デフォルト: 1時間
    ):
        """
        初期化メソッド

        Args:
            redis_service: Redisサービスインスタンス
            main_repository: メインのユーザーリポジトリ実装
            ttl_seconds: キャッシュの有効期限（秒）
        """
        self.redis = redis_service
        self.main_repository = main_repository
        self.ttl = ttl_seconds

    def _get_user_key(self, user_id: str) -> str:
        """
        ユーザーIDからRedisキーを生成します

        Args:
            user_id: ユーザーID

        Returns:
            Redisキー文字列
        """
        return f"{self.USER_KEY_PREFIX}{user_id}"

    def _get_email_key(self, email: str) -> str:
        """
        メールアドレスからインデックスキーを生成します

        Args:
            email: ユーザーのメールアドレス

        Returns:
            Redisキー文字列
        """
        return f"{self.EMAIL_INDEX_PREFIX}{email}"

    def _get_username_key(self, username: str) -> str:
        """
        ユーザー名からインデックスキーを生成します

        Args:
            username: ユーザー名

        Returns:
            Redisキー文字列
        """
        return f"{self.USERNAME_INDEX_PREFIX}{username}"

    def _serialize_user(self, user: User) -> Dict[str, Any]:
        """
        ユーザーオブジェクトを辞書に変換します

        Args:
            user: ユーザーオブジェクト

        Returns:
            ユーザーデータの辞書
        """
        return {
            "id": user.id,
            "email": user.email,
            "username": user.username if hasattr(user, "username") else None,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
            "credentials": {
                "hashed_password": user.credentials.password_hash,
                "is_active": user.credentials.is_active,
                "is_verified": user.credentials.is_verified,
                "mfa_enabled": user.credentials.mfa_enabled if hasattr(user.credentials, "mfa_enabled") else False,
                "mfa_type": user.credentials.mfa_type.value if hasattr(user.credentials, "mfa_type") else None
            },
            "profile": {
                "first_name": user.profile.first_name,
                "last_name": user.profile.last_name,
                "company_id": user.profile.company_id,
                "role": user.profile.role.value if isinstance(user.profile.role, UserRole) else user.profile.role
            }
        }

    def _deserialize_user(self, data: Dict[str, Any]) -> User:
        """
        辞書からユーザーオブジェクトを復元します

        Args:
            data: ユーザーデータの辞書

        Returns:
            ユーザーオブジェクト
        """
        credentials_data = data.get("credentials", {})
        profile_data = data.get("profile", {})

        # 日時文字列をdatetimeオブジェクトに変換
        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
        updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None

        # 必須のオブジェクトを構築
        credentials = UserCredentials(
            password_hash=credentials_data.get("hashed_password", ""),
            is_active=credentials_data.get("is_active", True),
            is_verified=credentials_data.get("is_verified", False)
        )

        profile = UserProfile(
            first_name=profile_data.get("first_name", ""),
            last_name=profile_data.get("last_name", ""),
            company_id=profile_data.get("company_id"),
            role=profile_data.get("role", "user")
        )

        # ユーザーオブジェクトを構築
        return User(
            id=data.get("id", ""),
            email=data.get("email", ""),
            username=data.get("username"),
            credentials=credentials,
            profile=profile,
            created_at=created_at,
            updated_at=updated_at
        )

    async def get_by_id(self, user_id: str) -> User:
        """
        IDによりユーザーを取得します。キャッシュにあればそこから、なければメインリポジトリから取得します。

        Args:
            user_id: 取得するユーザーのID

        Returns:
            ユーザーオブジェクト

        Raises:
            UserNotFoundError: ユーザーが見つからない場合
        """
        # キャッシュからの取得を試みる
        cache_key = self._get_user_key(user_id)
        cached_data = await self.redis.get_json(cache_key)

        if cached_data:
            # キャッシュヒット
            logger.debug(f"ユーザー(ID: {user_id})のキャッシュヒット")
            return self._deserialize_user(cached_data)

        # キャッシュミス - メインリポジトリから取得
        try:
            logger.debug(f"ユーザー(ID: {user_id})のキャッシュミス、メインリポジトリから取得")
            user = await self.main_repository.get_by_id(user_id)

            # キャッシュに保存
            user_data = self._serialize_user(user)
            await self.redis.set_json(cache_key, user_data, self.ttl)

            # インデックスの更新
            await self._update_indices(user)

            return user
        except UserNotFoundError:
            # メインリポジトリでも見つからない場合は例外を再発生
            logger.warning(f"ユーザー(ID: {user_id})が見つかりません")
            raise

    async def get_by_email(self, email: str) -> User:
        """
        メールアドレスによりユーザーを取得します。
        キャッシュにメールアドレスインデックスがあればそれを使用し、
        なければメインリポジトリから取得します。

        Args:
            email: ユーザーのメールアドレス

        Returns:
            ユーザーオブジェクト

        Raises:
            UserNotFoundError: ユーザーが見つからない場合
        """
        # メールアドレスインデックスの確認
        email_key = self._get_email_key(email)
        user_id = await self.redis.get_value(email_key)

        if user_id:
            # インデックスが見つかった場合はIDで取得を試みる
            try:
                return await self.get_by_id(user_id)
            except UserNotFoundError:
                # IDが無効な場合はインデックスを削除
                await self.redis.delete_key(email_key)

        # メインリポジトリから取得
        try:
            user = await self.main_repository.get_by_email(email)

            # ユーザーデータとインデックスをキャッシュに保存
            await self._cache_user_with_indices(user)

            return user
        except UserNotFoundError:
            logger.warning(f"メールアドレス '{email}' のユーザーが見つかりません")
            raise

    async def get_by_username(self, username: str) -> User:
        """
        ユーザー名によりユーザーを取得します。
        キャッシュにユーザー名インデックスがあればそれを使用し、
        なければメインリポジトリから取得します。

        Args:
            username: ユーザー名

        Returns:
            ユーザーオブジェクト

        Raises:
            UserNotFoundError: ユーザーが見つからない場合
        """
        # ユーザー名インデックスの確認
        username_key = self._get_username_key(username)
        user_id = await self.redis.get_value(username_key)

        if user_id:
            # インデックスが見つかった場合はIDで取得を試みる
            try:
                return await self.get_by_id(user_id)
            except UserNotFoundError:
                # IDが無効な場合はインデックスを削除
                await self.redis.delete_key(username_key)

        # メインリポジトリから取得
        try:
            user = await self.main_repository.get_by_username(username)

            # ユーザーデータとインデックスをキャッシュに保存
            await self._cache_user_with_indices(user)

            return user
        except UserNotFoundError:
            logger.warning(f"ユーザー名 '{username}' のユーザーが見つかりません")
            raise

    async def create(self, user: User) -> User:
        """
        新しいユーザーを作成します。メインリポジトリで作成し、キャッシュに保存します。

        Args:
            user: 作成するユーザーオブジェクト

        Returns:
            作成されたユーザーオブジェクト
        """
        # メインリポジトリでユーザーを作成
        created_user = await self.main_repository.create(user)
        logger.info(f"ユーザー(ID: {created_user.id})を作成しました")

        # ユーザーデータとインデックスをキャッシュに保存
        await self._cache_user_with_indices(created_user)

        return created_user

    async def update(self, user: User) -> User:
        """
        ユーザー情報を更新します。メインリポジトリで更新し、キャッシュも更新します。

        Args:
            user: 更新するユーザーオブジェクト

        Returns:
            更新されたユーザーオブジェクト

        Raises:
            UserNotFoundError: ユーザーが見つからない場合
        """
        # 現在のユーザー情報を取得（インデックス更新用）
        try:
            current_user = await self.get_by_id(user.id)
        except UserNotFoundError:
            current_user = None

        # メインリポジトリでユーザーを更新
        updated_user = await self.main_repository.update(user)
        logger.info(f"ユーザー(ID: {updated_user.id})を更新しました")

        # 古いインデックスがあれば削除
        if current_user:
            if current_user.email != updated_user.email:
                await self.redis.delete_key(self._get_email_key(current_user.email))

            if hasattr(current_user, "username") and hasattr(updated_user, "username"):
                if current_user.username != updated_user.username:
                    await self.redis.delete_key(self._get_username_key(current_user.username))

        # ユーザーデータとインデックスをキャッシュに保存
        await self._cache_user_with_indices(updated_user)

        return updated_user

    async def delete(self, user_id: str) -> None:
        """
        ユーザーを削除します。メインリポジトリから削除し、キャッシュからも削除します。

        Args:
            user_id: 削除するユーザーのID

        Raises:
            UserNotFoundError: ユーザーが見つからない場合
        """
        # 現在のユーザー情報を取得（インデックス削除用）
        try:
            user = await self.get_by_id(user_id)
        except UserNotFoundError:
            # メインリポジトリから削除を試みる
            await self.main_repository.delete(user_id)
            return

        # メインリポジトリからユーザーを削除
        await self.main_repository.delete(user_id)
        logger.info(f"ユーザー(ID: {user_id})を削除しました")

        # キャッシュからも削除
        await self.redis.delete_key(self._get_user_key(user_id))

        # インデックスも削除
        await self.redis.delete_key(self._get_email_key(user.email))
        if hasattr(user, "username") and user.username:
            await self.redis.delete_key(self._get_username_key(user.username))

    async def get_all(self) -> List[User]:
        """
        すべてのユーザーを取得します。
        (キャッシュは利用せずメインリポジトリから直接取得します)

        Returns:
            ユーザーオブジェクトのリスト
        """
        # すべてのユーザーの取得はキャッシュを使わない
        users = await self.main_repository.get_all()

        # 結果をキャッシュに保存（各ユーザーを個別にキャッシュ）
        for user in users:
            await self._cache_user_with_indices(user)

        return users

    async def _cache_user_with_indices(self, user: User) -> None:
        """
        ユーザーデータとインデックスをキャッシュに保存します

        Args:
            user: キャッシュするユーザー
        """
        # ユーザーデータをキャッシュ
        user_key = self._get_user_key(user.id)
        user_data = self._serialize_user(user)
        await self.redis.set_json(user_key, user_data, self.ttl)

        # インデックスを更新
        await self._update_indices(user)

    async def _update_indices(self, user: User) -> None:
        """
        ユーザーに関連するインデックスを更新します

        Args:
            user: インデックスを更新するユーザー
        """
        # メールアドレスインデックス
        email_key = self._get_email_key(user.email)
        await self.redis.set_value(email_key, user.id, self.ttl)

        # ユーザー名インデックス（存在する場合）
        if hasattr(user, "username") and user.username:
            username_key = self._get_username_key(user.username)
            await self.redis.set_value(username_key, user.id, self.ttl)


def create_redis_user_repository(
    redis_service: RedisService,
    main_repository: UserRepositoryInterface,
    ttl: int = 3600
) -> RedisUserRepository:
    """
    RedisUserRepositoryインスタンスを作成するファクトリ関数

    Args:
        redis_service: Redisサービスのインスタンス
        main_repository: メインのユーザーリポジトリ実装
        ttl: キャッシュの有効期限（秒）

    Returns:
        設定されたRedisUserRepositoryインスタンス
    """
    return RedisUserRepository(
        redis_service=redis_service,
        main_repository=main_repository,
        ttl_seconds=ttl
    )