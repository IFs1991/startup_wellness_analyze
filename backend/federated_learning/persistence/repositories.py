# Phase 3 Task 3.1: データベース層の実装
# TDD GREEN段階: リポジトリパターン実装

"""
リポジトリパターン実装モジュール

このモジュールは以下のリポジトリクラスを提供します：
- ModelRepository: フェデレーテッド学習モデルの管理
- ClientRegistryRepository: クライアント登録・管理
- TrainingHistoryRepository: 学習セッション履歴管理
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple, Union
from abc import ABC, abstractmethod

from sqlalchemy import select, update, delete, func, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import structlog

from .models import FLModel, ClientRegistration, TrainingSession, PrivacyBudgetRecord, SystemAuditLog
from .database import get_database_manager

logger = structlog.get_logger(__name__)


class BaseRepository(ABC):
    """リポジトリ基底クラス"""

    def __init__(self):
        self.db_manager = get_database_manager()

    async def get_session(self):
        """データベースセッションを取得"""
        return self.db_manager.get_session()


class ModelRepository(BaseRepository):
    """
    FLModelのリポジトリクラス

    機能:
    - モデルのCRUD操作
    - バージョン管理
    - 検索・フィルタリング
    - ページネーション
    """

    async def create(self, model_data: Dict[str, Any]) -> FLModel:
        """
        新しいモデルを作成

        Args:
            model_data: モデル作成データ

        Returns:
            FLModel: 作成されたモデルインスタンス

        Raises:
            IntegrityError: 重複するモデル名・バージョンの場合
            SQLAlchemyError: データベース操作エラー
        """
        try:
            async with self.db_manager.get_session() as session:
                # モデルサイズの計算
                model_bytes = model_data.get("model_bytes", b"")
                model_size = len(model_bytes)

                model = FLModel(
                    id=model_data.get("id", str(uuid.uuid4())),
                    name=model_data["name"],
                    version=model_data["version"],
                    description=model_data.get("description"),
                    model_bytes=model_bytes,
                    model_size=model_size,
                    model_metadata=model_data.get("metadata", {}),
                    accuracy=str(model_data.get("accuracy", "")),
                    loss=str(model_data.get("loss", "")),
                    privacy_epsilon=str(model_data.get("privacy_epsilon", "")),
                    privacy_delta=str(model_data.get("privacy_delta", "")),
                    status=model_data.get("status", "training"),
                    created_at=model_data.get("created_at", datetime.now(timezone.utc))
                )

                session.add(model)
                await session.commit()
                await session.refresh(model)

                logger.info(f"Created model: {model.name} v{model.version}")
                return model

        except IntegrityError as e:
            logger.error(f"Model creation failed - duplicate name/version: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error during model creation: {e}")
            raise

    async def get_by_id(self, model_id: str) -> Optional[FLModel]:
        """IDでモデルを取得"""
        async with self.db_manager.get_session() as session:
            stmt = select(FLModel).where(FLModel.id == model_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_latest_version(self, model_name: str) -> Optional[FLModel]:
        """指定したモデル名の最新バージョンを取得"""
        async with self.db_manager.get_session() as session:
            stmt = (
                select(FLModel)
                .where(FLModel.name == model_name)
                .where(FLModel.is_active == True)
                .order_by(desc(FLModel.created_at))
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def list_models(
        self,
        offset: int = 0,
        limit: int = 10,
        status_filter: Optional[str] = None,
        name_filter: Optional[str] = None
    ) -> Tuple[List[FLModel], int]:
        """モデル一覧を取得（ページネーション付き）"""
        async with self.db_manager.get_session() as session:
            # ベースクエリ
            stmt = select(FLModel).where(FLModel.is_active == True)
            count_stmt = select(func.count(FLModel.id)).where(FLModel.is_active == True)

            # フィルタリング
            if status_filter:
                stmt = stmt.where(FLModel.status == status_filter)
                count_stmt = count_stmt.where(FLModel.status == status_filter)

            if name_filter:
                stmt = stmt.where(FLModel.name.ilike(f"%{name_filter}%"))
                count_stmt = count_stmt.where(FLModel.name.ilike(f"%{name_filter}%"))

            # ソートとページネーション
            stmt = stmt.order_by(desc(FLModel.created_at)).offset(offset).limit(limit)

            # 実行
            models_result = await session.execute(stmt)
            count_result = await session.execute(count_stmt)

            models = models_result.scalars().all()
            total_count = count_result.scalar()

            return list(models), total_count

    async def update_model(self, model_id: str, update_data: Dict[str, Any]) -> Optional[FLModel]:
        """モデル情報を更新"""
        async with self.db_manager.get_session() as session:
            stmt = (
                update(FLModel)
                .where(FLModel.id == model_id)
                .values(**update_data)
                .returning(FLModel)
            )
            result = await session.execute(stmt)
            await session.commit()

            updated_model = result.scalar_one_or_none()
            if updated_model:
                logger.info(f"Updated model: {model_id}")

            return updated_model

    async def soft_delete(self, model_id: str) -> bool:
        """モデルをソフト削除"""
        async with self.db_manager.get_session() as session:
            stmt = (
                update(FLModel)
                .where(FLModel.id == model_id)
                .values(is_active=False, updated_at=datetime.now(timezone.utc))
            )
            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0

    async def get_models_by_name(self, model_name: str) -> List[FLModel]:
        """モデル名で全バージョンを取得"""
        async with self.db_manager.get_session() as session:
            stmt = (
                select(FLModel)
                .where(FLModel.name == model_name)
                .where(FLModel.is_active == True)
                .order_by(desc(FLModel.created_at))
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())


class ClientRegistryRepository(BaseRepository):
    """
    ClientRegistrationのリポジトリクラス

    機能:
    - クライアント登録・管理
    - ステータス管理
    - 能力ベースフィルタリング
    - 統計情報管理
    """

    async def register(self, client_data: Dict[str, Any]) -> ClientRegistration:
        """新しいクライアントを登録"""
        async with self.db_manager.get_session() as session:
            client = ClientRegistration(
                client_id=client_data["client_id"],
                public_key=client_data["public_key"],
                certificate=client_data["certificate"],
                certificate_fingerprint=self._generate_fingerprint(client_data["certificate"]),
                capabilities=client_data.get("capabilities", {}),
                status=client_data.get("status", "active"),
                last_seen_ip=client_data.get("last_seen_ip"),
                user_agent=client_data.get("user_agent"),
                registration_source=client_data.get("registration_source", "manual")
            )

            session.add(client)
            await session.commit()
            await session.refresh(client)

            logger.info(f"Registered client: {client.client_id}")
            return client

    def _generate_fingerprint(self, certificate: str) -> str:
        """証明書フィンガープリントを生成"""
        import hashlib
        return hashlib.sha256(certificate.encode()).hexdigest()

    async def get_by_id(self, client_id: str) -> Optional[ClientRegistration]:
        """クライアントIDで取得"""
        async with self.db_manager.get_session() as session:
            stmt = select(ClientRegistration).where(ClientRegistration.client_id == client_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_active_clients(self) -> List[ClientRegistration]:
        """アクティブなクライアント一覧を取得"""
        async with self.db_manager.get_session() as session:
            stmt = (
                select(ClientRegistration)
                .where(ClientRegistration.status == "active")
                .order_by(desc(ClientRegistration.last_active_at))
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def update_status(self, client_id: str, new_status: str) -> bool:
        """クライアントステータスを更新"""
        async with self.db_manager.get_session() as session:
            stmt = (
                update(ClientRegistration)
                .where(ClientRegistration.client_id == client_id)
                .values(
                    status=new_status,
                    updated_at=datetime.now(timezone.utc)
                )
            )
            result = await session.execute(stmt)
            await session.commit()

            success = result.rowcount > 0
            if success:
                logger.info(f"Updated client {client_id} status to {new_status}")

            return success

    async def update_last_active(self, client_id: str, ip_address: Optional[str] = None) -> bool:
        """クライアントの最終アクティブ時刻を更新"""
        async with self.db_manager.get_session() as session:
            update_data = {
                "last_active_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            if ip_address:
                update_data["last_seen_ip"] = ip_address

            stmt = (
                update(ClientRegistration)
                .where(ClientRegistration.client_id == client_id)
                .values(**update_data)
            )
            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0

    async def get_clients_by_capabilities(self, required_capabilities: Dict[str, Any]) -> List[ClientRegistration]:
        """能力要件に基づいてクライアントを検索"""
        async with self.db_manager.get_session() as session:
            stmt = (
                select(ClientRegistration)
                .where(ClientRegistration.status == "active")
            )

            # JSON フィールドでの検索（簡易実装）
            # 実際のプロダクションではより詳細な検索条件が必要
            result = await session.execute(stmt)
            all_clients = result.scalars().all()

            # Pythonレベルでフィルタリング
            filtered_clients = []
            for client in all_clients:
                if self._check_capabilities(client.capabilities, required_capabilities):
                    filtered_clients.append(client)

            return filtered_clients

    def _check_capabilities(self, client_caps: Dict, required_caps: Dict) -> bool:
        """能力要件をチェック"""
        for key, required_value in required_caps.items():
            client_value = client_caps.get(key)
            if client_value is None:
                return False

            # 数値比較の場合
            if isinstance(required_value, (int, float)):
                if client_value < required_value:
                    return False
            # ブール値の場合
            elif isinstance(required_value, bool):
                if client_value != required_value:
                    return False
            # 文字列の場合
            else:
                if str(client_value) != str(required_value):
                    return False

        return True

    async def increment_session_count(self, client_id: str) -> bool:
        """セッション参加回数をインクリメント"""
        async with self.db_manager.get_session() as session:
            stmt = (
                update(ClientRegistration)
                .where(ClientRegistration.client_id == client_id)
                .values(
                    total_sessions=ClientRegistration.total_sessions + 1,
                    updated_at=datetime.now(timezone.utc)
                )
            )
            result = await session.execute(stmt)
            await session.commit()

            return result.rowcount > 0


class TrainingHistoryRepository(BaseRepository):
    """
    TrainingSessionのリポジトリクラス

    機能:
    - 学習セッション管理
    - 履歴追跡
    - パフォーマンス分析
    - レポート生成
    """

    async def create_session(self, session_data: Dict[str, Any]) -> TrainingSession:
        """新しい学習セッションを作成"""
        async with self.db_manager.get_session() as session:
            training_session = TrainingSession(
                session_id=session_data.get("session_id", str(uuid.uuid4())),
                model_id=session_data["model_id"],
                round_number=session_data["round_number"],
                session_name=session_data.get("session_name"),
                participating_clients=session_data.get("participating_clients", []),
                selected_clients=session_data.get("selected_clients", []),
                aggregation_result=session_data.get("aggregation_result", {}),
                privacy_metrics=session_data.get("privacy_metrics", {}),
                started_at=session_data.get("started_at", datetime.now(timezone.utc))
            )

            session.add(training_session)
            await session.commit()
            await session.refresh(training_session)

            logger.info(f"Created training session: {training_session.session_id}")
            return training_session

    async def get_by_id(self, session_id: str) -> Optional[TrainingSession]:
        """セッションIDで取得"""
        async with self.db_manager.get_session() as session:
            stmt = (
                select(TrainingSession)
                .options(selectinload(TrainingSession.model))
                .where(TrainingSession.session_id == session_id)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_by_model_id(self, model_id: str) -> List[TrainingSession]:
        """モデル別学習履歴を取得"""
        async with self.db_manager.get_session() as session:
            stmt = (
                select(TrainingSession)
                .where(TrainingSession.model_id == model_id)
                .order_by(asc(TrainingSession.round_number))
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_recent_sessions(self, limit: int = 10) -> List[TrainingSession]:
        """最近の学習セッションを取得"""
        async with self.db_manager.get_session() as session:
            stmt = (
                select(TrainingSession)
                .options(selectinload(TrainingSession.model))
                .order_by(desc(TrainingSession.started_at))
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def update_session_status(
        self,
        session_id: str,
        status: str,
        completion_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """セッションステータスを更新"""
        async with self.db_manager.get_session() as session:
            update_data = {"status": status}

            if status == "completed" and completion_data:
                update_data.update({
                    "completed_at": datetime.now(timezone.utc),
                    "aggregation_result": completion_data.get("aggregation_result", {}),
                    "accuracy_after": str(completion_data.get("accuracy_after", "")),
                    "loss_after": str(completion_data.get("loss_after", "")),
                    "total_updates_received": completion_data.get("total_updates_received", 0),
                    "session_duration": completion_data.get("session_duration")
                })
            elif status == "failed" and completion_data:
                update_data.update({
                    "completed_at": datetime.now(timezone.utc),
                    "error_message": completion_data.get("error_message"),
                    "failed_clients": completion_data.get("failed_clients", [])
                })

            stmt = (
                update(TrainingSession)
                .where(TrainingSession.session_id == session_id)
                .values(**update_data)
            )
            result = await session.execute(stmt)
            await session.commit()

            success = result.rowcount > 0
            if success:
                logger.info(f"Updated session {session_id} status to {status}")

            return success

    async def get_session_statistics(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """セッション統計を取得"""
        async with self.db_manager.get_session() as session:
            base_stmt = select(TrainingSession)
            if model_id:
                base_stmt = base_stmt.where(TrainingSession.model_id == model_id)

            # 総セッション数
            total_count_stmt = select(func.count(TrainingSession.session_id)).select_from(base_stmt.subquery())
            total_count = await session.execute(total_count_stmt)

            # 完了セッション数
            completed_stmt = base_stmt.where(TrainingSession.status == "completed")
            completed_count_stmt = select(func.count(TrainingSession.session_id)).select_from(completed_stmt.subquery())
            completed_count = await session.execute(completed_count_stmt)

            # 失敗セッション数
            failed_stmt = base_stmt.where(TrainingSession.status == "failed")
            failed_count_stmt = select(func.count(TrainingSession.session_id)).select_from(failed_stmt.subquery())
            failed_count = await session.execute(failed_count_stmt)

            return {
                "total_sessions": total_count.scalar(),
                "completed_sessions": completed_count.scalar(),
                "failed_sessions": failed_count.scalar(),
                "success_rate": (completed_count.scalar() / max(total_count.scalar(), 1)) * 100
            }