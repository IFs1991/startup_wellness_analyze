import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime
import json
from google.cloud import firestore

from core.text_miner import TextMiner, TextAnalysisError, AnalysisResult

@pytest.fixture
def sample_text_data():
    """テスト用のテキストデータを提供します"""
    return (
        "当社の製品に対する顧客満足度は非常に高いです。特に操作性とデザインについて高評価を得ています。"
        "ただし、価格についてはやや高いという意見もあります。競合他社と比較すると、品質は上回っていますが、"
        "アフターサポートについては改善の余地があります。次回のアップデートでは、ユーザーインターフェースの"
        "さらなる改善と、新機能の追加を予定しています。"
    )

@pytest.mark.asyncio
@patch('core.text_miner.genai.GenerativeModel')
@patch('google.cloud.firestore.Client')
async def test_analyze_text(mock_firestore_client, mock_generative_model):
    """テキスト分析機能をテスト"""
    # Firestoreモックの設定
    firestore_mock = MagicMock()
    collection_mock = MagicMock()
    doc_mock = MagicMock()
    mock_firestore_client.return_value = firestore_mock
    firestore_mock.collection.return_value = collection_mock
    collection_mock.document.return_value = doc_mock

    # Gemini APIモックの設定
    model_instance = AsyncMock()
    mock_generative_model.return_value = model_instance

    # 生成AIの回答をモック
    ai_response = AsyncMock()
    ai_response.text = json.dumps({
        "sentiment_score": 0.7,
        "keywords": ["顧客満足度", "操作性", "デザイン", "価格", "アフターサポート"],
        "categories": ["製品評価", "顧客フィードバック"],
        "summary": "製品の操作性とデザインは高評価だが、価格とアフターサポートに改善の余地あり",
        "main_topics": ["顧客満足度", "製品品質", "価格", "サポート"],
        "word_count": 20
    })
    model_instance.generate_content.return_value = ai_response

    # TextMinerインスタンスの作成
    text_miner = TextMiner(firestore_mock, "test_api_key")

    # テスト用テキストデータ
    sample_text = sample_text_data()

    # メタデータ
    metadata = {
        "source": "顧客アンケート",
        "collection_date": "2023-01-15",
        "category": "製品フィードバック"
    }

    # テキスト分析の実行
    result = await text_miner.analyze_text(
        text=sample_text,
        user_id="test_user_id",
        source_id="survey_123",
        metadata=metadata
    )

    # 結果の検証
    assert result is not None
    assert result["sentiment_score"] == 0.7
    assert "keywords" in result
    assert "summary" in result
    assert len(result["keywords"]) > 0

    # Gemini APIが呼ばれたことを確認
    model_instance.generate_content.assert_called_once()

    # Firestoreへの保存が呼ばれたことを確認
    collection_mock.document.assert_called_once()
    doc_mock.set.assert_called_once()
    set_args = doc_mock.set.call_args[0][0]
    assert set_args["user_id"] == "test_user_id"
    assert set_args["source_id"] == "survey_123"
    assert set_args["metadata"] == metadata
    assert "created_at" in set_args
    assert "analysis_result" in set_args

@pytest.mark.asyncio
@patch('core.text_miner.genai.GenerativeModel')
@patch('google.cloud.firestore.Client')
async def test_analyze_text_error_handling(mock_firestore_client, mock_generative_model):
    """テキスト分析のエラーハンドリングをテスト"""
    # Firestoreモックの設定
    firestore_mock = MagicMock()
    mock_firestore_client.return_value = firestore_mock

    # Gemini APIモックの設定 - エラーを発生させる
    model_instance = AsyncMock()
    mock_generative_model.return_value = model_instance
    model_instance.generate_content.side_effect = Exception("API error")

    # TextMinerインスタンスの作成
    text_miner = TextMiner(firestore_mock, "test_api_key")

    # テスト用テキストデータ
    sample_text = sample_text_data()

    # エラーが発生することを確認
    with pytest.raises(TextAnalysisError) as excinfo:
        await text_miner.analyze_text(
            text=sample_text,
            user_id="test_user_id"
        )

    # エラーメッセージの検証
    assert "テキスト分析に失敗しました" in str(excinfo.value)

@pytest.mark.asyncio
@patch('google.cloud.firestore.Client')
async def test_get_analysis_history(mock_firestore_client):
    """分析履歴の取得をテスト"""
    # Firestoreモックの設定
    firestore_mock = MagicMock()
    collection_mock = MagicMock()
    query_mock = MagicMock()
    mock_firestore_client.return_value = firestore_mock
    firestore_mock.collection.return_value = collection_mock
    collection_mock.where.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.limit.return_value = query_mock

    # クエリ結果のモック
    mock_docs = []
    for i in range(3):
        doc_mock = MagicMock()
        doc_mock.id = f"analysis{i}"
        doc_mock.to_dict.return_value = {
            "user_id": "test_user_id",
            "created_at": datetime.now(),
            "text_snippet": f"テキストサンプル{i}",
            "analysis_result": {
                "sentiment_score": 0.5 + i * 0.1,
                "keywords": [f"キーワード{j}" for j in range(3)],
                "summary": f"要約文{i}"
            },
            "metadata": {
                "source": "顧客アンケート"
            }
        }
        mock_docs.append(doc_mock)

    # クエリの実行結果を設定
    query_mock.stream.return_value = mock_docs

    # TextMinerインスタンスの作成
    text_miner = TextMiner(firestore_mock, "test_api_key")

    # 分析履歴の取得
    history = await text_miner.get_analysis_history(user_id="test_user_id", limit=10)

    # 結果の検証
    assert history is not None
    assert len(history) == 3
    assert history[0]["id"] == "analysis0"
    assert "sentiment_score" in history[0]["analysis_result"]
    assert "keywords" in history[0]["analysis_result"]
    assert "summary" in history[0]["analysis_result"]

    # Firestoreクエリーが正しく呼び出されたことを確認
    collection_mock.where.assert_called_once_with("user_id", "==", "test_user_id")
    query_mock.order_by.assert_called_once_with("created_at", direction="DESCENDING")
    query_mock.limit.assert_called_once_with(10)
    query_mock.stream.assert_called_once()

@pytest.mark.asyncio
@patch('google.cloud.firestore.Client')
async def test_result_conversion(mock_firestore_client):
    """AnalysisResultの変換をテスト"""
    # AnalysisResultのインスタンス作成
    analysis_result = AnalysisResult(
        sentiment_score=0.8,
        keywords=["キーワード1", "キーワード2", "キーワード3"],
        categories=["カテゴリA", "カテゴリB"],
        summary="テストの要約文",
        main_topics=["トピックX", "トピックY"],
        word_count=150
    )

    # 辞書への変換
    result_dict = analysis_result.to_dict()

    # 結果の検証
    assert result_dict["sentiment_score"] == 0.8
    assert len(result_dict["keywords"]) == 3
    assert "キーワード1" in result_dict["keywords"]
    assert len(result_dict["categories"]) == 2
    assert "カテゴリA" in result_dict["categories"]
    assert result_dict["summary"] == "テストの要約文"
    assert len(result_dict["main_topics"]) == 2
    assert "トピックX" in result_dict["main_topics"]
    assert result_dict["statistics"]["word_count"] == 150