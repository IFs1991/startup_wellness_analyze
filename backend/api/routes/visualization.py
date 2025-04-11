"""
このモジュールは非推奨となりました。
代わりに`backend.api.routers.visualization`を使用してください。
このファイルは後方互換性のために残されています。
"""

import logging
import warnings
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.routing import APIRoute

logger = logging.getLogger(__name__)
warnings.warn(
    "routes.visualization モジュールは非推奨です。代わりに routers.visualization を使用してください。",
    DeprecationWarning,
    stacklevel=2
)

# 元のルーター
router = APIRouter(prefix="/api/v1/visualizations", tags=["visualizations"])

# リダイレクト処理を行うクラス
class RedirectRoute(APIRoute):
    def __init__(self, *args, **kwargs):
        self.redirect_path_prefix = kwargs.pop("redirect_path_prefix", "/api/visualization")
        super().__init__(*args, **kwargs)

    async def handle(self, request: Request) -> RedirectResponse:
        # 元のパスからプレフィックスを除去し、新しいパスを作成
        path = request.url.path
        new_path = path.replace("/api/v1/visualizations", self.redirect_path_prefix)

        # クエリパラメータを維持
        if request.url.query:
            new_path = f"{new_path}?{request.url.query}"

        logger.info(f"可視化リクエストをリダイレクト: {path} -> {new_path}")
        return RedirectResponse(url=new_path)

# 元のエンドポイントパターンに対応するリダイレクトルートを作成
@router.api_route("/chart", methods=["POST"], response_class=RedirectResponse, route_class=RedirectRoute)
async def redirect_generate_chart(request: Request):
    return {}

@router.api_route("/multiple-charts", methods=["POST"], response_class=RedirectResponse, route_class=RedirectRoute)
async def redirect_generate_multiple_charts(request: Request):
    return {}

@router.api_route("/dashboard", methods=["POST"], response_class=RedirectResponse, route_class=RedirectRoute)
async def redirect_generate_dashboard(request: Request):
    return {}

@router.api_route("/chart/background", methods=["POST"], response_class=RedirectResponse, route_class=RedirectRoute)
async def redirect_generate_chart_background(request: Request):
    return {}

@router.api_route("/chart/status/{cache_key}", methods=["GET"], response_class=RedirectResponse, route_class=RedirectRoute)
async def redirect_get_chart_status(request: Request, cache_key: str):
    return {}

# すべてのHTTPメソッドに対応する汎用的なキャッチオールルート
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"], response_class=RedirectResponse, route_class=RedirectRoute)
async def redirect_all(request: Request, path: str):
    return {}