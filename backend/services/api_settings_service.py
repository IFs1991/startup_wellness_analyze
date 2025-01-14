from typing import Optional
from ..models.api_settings import OpenAISettings, OpenAISettingsResponse
from google.cloud import secretmanager
import os
import json

class APISettingsService:
    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.secret_id = "openai_settings"
        self.client = secretmanager.SecretManagerServiceClient()

    async def save_openai_settings(self, settings: OpenAISettings) -> OpenAISettingsResponse:
        """OpenAI API設定を保存"""
        try:
            # APIキーを環境変数に設定
            os.environ["OPENAI_API_KEY"] = settings.api_key

            # 設定をJSON形式で保存
            settings_dict = settings.dict()
            settings_json = json.dumps(settings_dict)

            # Secret Managerに保存
            parent = f"projects/{self.project_id}"

            # シークレットが存在するか確認
            try:
                secret = self.client.get_secret(name=f"{parent}/secrets/{self.secret_id}")
            except Exception:
                # シークレットが存在しない場合は作成
                secret = self.client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": self.secret_id,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )

            # シークレットの新しいバージョンを作成
            secret_version = self.client.add_secret_version(
                request={
                    "parent": secret.name,
                    "payload": {"data": settings_json.encode("UTF-8")},
                }
            )

            # APIキーを除いた設定を返す
            return OpenAISettingsResponse(
                model=settings.model,
                request_limit=settings.request_limit,
                limit_period=settings.limit_period,
                notify_on_limit=settings.notify_on_limit
            )

        except Exception as e:
            raise Exception(f"設定の保存に失敗しました: {str(e)}")

    async def get_openai_settings(self) -> Optional[OpenAISettingsResponse]:
        """OpenAI API設定を取得"""
        try:
            # Secret Managerから設定を取得
            name = f"projects/{self.project_id}/secrets/{self.secret_id}/versions/latest"
            response = self.client.access_secret_version(request={"name": name})
            settings_json = response.payload.data.decode("UTF-8")
            settings_dict = json.loads(settings_json)

            # APIキーを除いた設定を返す
            return OpenAISettingsResponse(
                model=settings_dict["model"],
                request_limit=settings_dict.get("request_limit", 1000),
                limit_period=settings_dict.get("limit_period", "hour"),
                notify_on_limit=settings_dict.get("notify_on_limit", True)
            )

        except Exception as e:
            return None