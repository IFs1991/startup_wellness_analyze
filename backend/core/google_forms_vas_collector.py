# -*- coding: utf-8 -*-
"""
VASデータ収集モジュール
Google Forms からVASデータを収集し、Firestoreに保存します。
"""
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datetime import datetime
import logging
import asyncio
from pydantic import BaseModel, Field
from backend.core.google_forms_connector import GoogleFormsConnector, FormResponse
from backend.service.firestore.client import FirestoreService

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class VASCollectionError(Exception):
    """VASデータ収集に関するエラー"""
    pass

class VASFormConfig(BaseModel):
    """VASフォーム設定"""
    form_id: str
    company_id_question: str
    category_mappings: Dict[str, str]  # Formの質問ID -> カテゴリ名
    additional_fields: Optional[Dict[str, str]] = None  # 追加フィールドのマッピング

class VASDataCollector:
    """
    Google FormsからVASデータを収集し、Firestoreに保存するクラス
    """
    def __init__(
        self,
        google_forms_connector: GoogleFormsConnector,
        firestore_service: FirestoreService
    ):
        """
        初期化メソッド

        Args:
            google_forms_connector: Google Forms API接続クラス
            firestore_service: Firestoreサービスクラス
        """
        self.forms_connector = google_forms_connector
        self.firestore_service = firestore_service
        logger.info("VASDataCollector initialized successfully")

    async def collect_vas_data(
        self,
        form_config: VASFormConfig,
        page_size: Optional[int] = None,
        last_collected_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        VASデータを収集してFirestoreに保存

        Args:
            form_config: VASフォームの設定
            page_size: 一度に取得するレスポンス数
            last_collected_time: 前回のデータ収集時刻

        Returns:
            Dict[str, Any]: 収集結果の要約
        """
        try:
            # フォームのレスポンスを取得
            responses = await self.forms_connector.get_form_responses(
                form_id=form_config.form_id,
                page_size=page_size
            )

            if not responses:
                logger.info(f"No responses found for form_id: {form_config.form_id}")
                return {"collected": 0, "status": "success", "message": "No new responses"}

            # 前回収集時以降のレスポンスのみをフィルタリング
            if last_collected_time:
                responses = [
                    r for r in responses
                    if r.last_submitted_time > last_collected_time
                ]

            if not responses:
                logger.info(f"No new responses since {last_collected_time}")
                return {"collected": 0, "status": "success", "message": "No new responses since last collection"}

            # レスポンスをVASデータに変換してFirestoreに保存
            collected_count = 0
            for response in responses:
                vas_data = self._convert_to_vas_data(response, form_config)
                if vas_data:
                    # Firestoreに保存
                    await self.firestore_service.add_document(
                        collection="vas_data",
                        data=vas_data
                    )
                    collected_count += 1

            logger.info(f"Successfully collected {collected_count} VAS responses")
            return {
                "collected": collected_count,
                "status": "success",
                "message": f"Successfully collected {collected_count} responses",
                "last_collection_time": datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"Error collecting VAS data: {str(e)}"
            logger.error(error_msg)
            return {
                "collected": 0,
                "status": "error",
                "message": error_msg,
                "last_collection_time": None
            }

    def _convert_to_vas_data(
        self,
        form_response: FormResponse,
        form_config: VASFormConfig
    ) -> Optional[Dict[str, Any]]:
        """
        フォームレスポンスをVASデータに変換

        Args:
            form_response: フォームレスポンス
            form_config: フォーム設定

        Returns:
            Optional[Dict[str, Any]]: VASデータ（変換できない場合はNone）
        """
        try:
            answers = form_response.answers

            # 企業IDを取得
            if form_config.company_id_question not in answers:
                logger.warning(f"Company ID question not found in response: {form_response.response_id}")
                return None

            company_id = answers[form_config.company_id_question]

            # VASカテゴリデータを抽出
            vas_data = {
                "company_id": company_id,
                "response_id": form_response.response_id,
                "timestamp": form_response.last_submitted_time,
                "source": f"google_forms:{form_config.form_id}"
            }

            # カテゴリスコアを抽出
            for question_id, category in form_config.category_mappings.items():
                if question_id in answers:
                    try:
                        # スコアを数値に変換（回答形式によって処理を分ける）
                        score_value = self._parse_score_value(answers[question_id])
                        vas_data[category] = score_value
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid score value for {category}: {answers[question_id]}")

            # 追加フィールドを抽出
            if form_config.additional_fields:
                for question_id, field_name in form_config.additional_fields.items():
                    if question_id in answers:
                        vas_data[field_name] = answers[question_id]

            # 総合スコアの計算（カテゴリスコアの平均）
            category_scores = [
                vas_data[category]
                for category in form_config.category_mappings.values()
                if category in vas_data
            ]

            if category_scores:
                vas_data["overall_score"] = sum(category_scores) / len(category_scores)

            return vas_data

        except Exception as e:
            logger.error(f"Error converting form response to VAS data: {str(e)}")
            return None

    def _parse_score_value(self, value: Any) -> float:
        """
        スコア値を数値に変換

        Args:
            value: 変換対象の値

        Returns:
            float: 変換後の数値
        """
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # "7/10" 形式の場合
            if '/' in value:
                numerator, denominator = value.split('/')
                return (float(numerator.strip()) / float(denominator.strip())) * 100
            # "70%" 形式の場合
            elif '%' in value:
                return float(value.replace('%', '').strip())
            # "7" 形式の場合（10段階評価と仮定）
            else:
                try:
                    score = float(value.strip())
                    # 10段階評価の場合は100段階に変換
                    if score <= 10:
                        return score * 10
                    return score
                except ValueError:
                    # 数値に変換できない場合はエラー
                    raise ValueError(f"Cannot convert to numeric value: {value}")
        else:
            raise TypeError(f"Unsupported value type: {type(value)}")


async def collect_vas_data_from_forms(
    service_account_file: str,
    form_configs: List[VASFormConfig]
) -> Dict[str, Any]:
    """
    複数のフォームからVASデータを収集する便利関数

    Args:
        service_account_file: サービスアカウントキーファイルのパス
        form_configs: フォーム設定のリスト

    Returns:
        Dict[str, Any]: 収集結果
    """
    from backend.core.google_forms_connector import create_forms_connector
    from backend.service.firestore.client import get_firestore_service

    # 必要なコネクタを作成
    forms_connector = create_forms_connector(service_account_file)
    firestore_service = get_firestore_service()

    # VASデータコレクタを作成
    collector = VASDataCollector(forms_connector, firestore_service)

    # 各フォームからデータを収集
    results = {}
    for config in form_configs:
        # 前回の収集時刻を取得
        collection_state = await firestore_service.get_document(
            collection="collection_state",
            document_id=f"vas_form_{config.form_id}"
        )

        last_collected_time = None
        if collection_state and 'last_collection_time' in collection_state:
            try:
                last_collected_time = datetime.fromisoformat(collection_state['last_collection_time'])
            except (ValueError, TypeError):
                pass

        # データ収集
        result = await collector.collect_vas_data(
            form_config=config,
            last_collected_time=last_collected_time
        )

        # 収集状態を更新
        if result['status'] == 'success' and result['collected'] > 0:
            await firestore_service.set_document(
                collection="collection_state",
                document_id=f"vas_form_{config.form_id}",
                data={
                    'form_id': config.form_id,
                    'last_collection_time': result['last_collection_time'],
                    'last_run_status': 'success',
                    'last_run_time': datetime.now().isoformat()
                }
            )

        results[config.form_id] = result

    # コネクタを閉じる
    await forms_connector.close()

    return results