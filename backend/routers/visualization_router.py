from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from datetime import datetime

from auth import get_current_user
from schemas import DashboardConfig, GraphConfig, VisualizationResponse
from src.database.firestore.client import FirestoreClient

router = APIRouter()
firestore_client = FirestoreClient()

class VisualizationService:
    """可視化サービスクラス"""
    def __init__(self, db_client):
        self.db_client = db_client

    async def create_visualization(
        self,
        config: Dict[str, Any],
        visualization_type: str,
        user_id: str
    ) -> Dict[str, Any]:
        """新しい可視化を作成する"""
        try:
            # 可視化IDの生成
            visualization_id = f"{visualization_type}_{datetime.utcnow().timestamp()}"

            # メタデータの設定
            visualization_data = {
                "id": visualization_id,
                "type": visualization_type,
                "config": config,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # データベースに保存
            await self.db_client.create_document(
                collection='visualizations',
                doc_id=visualization_id,
                data=visualization_data
            )

            return visualization_data
        except Exception as e:
            raise Exception(f"可視化の作成に失敗しました: {str(e)}")

    async def get_user_visualizations(self, user_id: str) -> List[Dict[str, Any]]:
        """ユーザーの可視化一覧を取得する"""
        try:
            visualizations = await self.db_client.query_documents(
                collection='visualizations',
                filters=[("user_id", "==", user_id)],
                order_by=('created_at', 'desc')
            )

            return visualizations
        except Exception as e:
            raise Exception(f"可視化一覧の取得に失敗しました: {str(e)}")

    async def get_visualization(self, visualization_id: str) -> Dict[str, Any]:
        """特定の可視化を取得する"""
        try:
            visualization = await self.db_client.get_document(
                collection='visualizations',
                doc_id=visualization_id
            )

            if not visualization:
                raise Exception("指定された可視化が見つかりません")

            return visualization
        except Exception as e:
            raise Exception(f"可視化の取得に失敗しました: {str(e)}")

    async def update_visualization(
        self,
        visualization_id: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """可視化を更新する"""
        try:
            update_data["updated_at"] = datetime.utcnow()

            await self.db_client.update_document(
                collection='visualizations',
                doc_id=visualization_id,
                data=update_data
            )

            # 更新後のデータを取得
            updated_visualization = await self.db_client.get_document(
                collection='visualizations',
                doc_id=visualization_id
            )

            return updated_visualization
        except Exception as e:
            raise Exception(f"可視化の更新に失敗しました: {str(e)}")

    async def delete_visualization(self, visualization_id: str) -> None:
        """可視化を削除する"""
        try:
            await self.db_client.delete_document(
                collection='visualizations',
                doc_id=visualization_id
            )
        except Exception as e:
            raise Exception(f"可視化の削除に失敗しました: {str(e)}")


def get_visualization_service() -> VisualizationService:
    """依存性注入: 可視化サービスの取得"""
    return VisualizationService(firestore_client)


@router.post("/dashboard/create", response_model=Dict[str, Any])
async def create_dashboard(
    config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    visualization_service: VisualizationService = Depends(get_visualization_service)
) -> Dict[str, Any]:
    """新しいダッシュボードを作成"""
    try:
        return await visualization_service.create_visualization(
            config=config,
            visualization_type="dashboard",
            user_id=current_user.get("uid")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/graph/create", response_model=Dict[str, Any])
async def create_graph(
    config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    visualization_service: VisualizationService = Depends(get_visualization_service)
) -> Dict[str, Any]:
    """新しいグラフを作成"""
    try:
        return await visualization_service.create_visualization(
            config=config,
            visualization_type="graph",
            user_id=current_user.get("uid")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/user", response_model=List[Dict[str, Any]])
async def get_user_visualizations(
    current_user = Depends(get_current_user),
    visualization_service: VisualizationService = Depends(get_visualization_service)
) -> List[Dict[str, Any]]:
    """ユーザーの可視化一覧を取得"""
    try:
        return await visualization_service.get_user_visualizations(
            user_id=current_user.get("uid")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{visualization_id}", response_model=Dict[str, Any])
async def get_visualization(
    visualization_id: str,
    current_user = Depends(get_current_user),
    visualization_service: VisualizationService = Depends(get_visualization_service)
) -> Dict[str, Any]:
    """可視化の詳細を取得"""
    try:
        visualization = await visualization_service.get_visualization(
            visualization_id=visualization_id
        )

        # 権限チェック
        if visualization.get("user_id") != current_user.get("uid"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="この可視化にアクセスする権限がありません"
            )

        return visualization
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.put("/{visualization_id}", response_model=Dict[str, Any])
async def update_visualization(
    visualization_id: str,
    update_data: Dict[str, Any],
    current_user = Depends(get_current_user),
    visualization_service: VisualizationService = Depends(get_visualization_service)
) -> Dict[str, Any]:
    """可視化を更新"""
    try:
        # 既存の可視化を取得して権限チェック
        visualization = await visualization_service.get_visualization(
            visualization_id=visualization_id
        )

        if visualization.get("user_id") != current_user.get("uid"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="この可視化を更新する権限がありません"
            )

        # 更新を実行
        return await visualization_service.update_visualization(
            visualization_id=visualization_id,
            update_data=update_data
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{visualization_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_visualization(
    visualization_id: str,
    current_user = Depends(get_current_user),
    visualization_service: VisualizationService = Depends(get_visualization_service)
) -> None:
    """可視化を削除"""
    try:
        # 既存の可視化を取得して権限チェック
        visualization = await visualization_service.get_visualization(
            visualization_id=visualization_id
        )

        if visualization.get("user_id") != current_user.get("uid"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="この可視化を削除する権限がありません"
            )

        # 削除を実行
        await visualization_service.delete_visualization(
            visualization_id=visualization_id
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )