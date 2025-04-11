"""
RedisWellnessRepositoryのテスト

Redisウェルネスリポジトリの機能をテストします。
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from backend.domain.models.wellness import (
    RecommendationAction,
    RecommendationPlan,
    ScoreCategory,
    ScoreHistory,
    ScoreMetric,
    WellnessScore,
)
from backend.domain.repositories.wellness_repository import WellnessRepositoryInterface
from backend.infrastructure.redis.redis_wellness_repository import RedisWellnessRepository

@pytest.fixture
def mock_redis_service():
    """Redisサービスのモックを作成します"""
    service = AsyncMock()
    return service

@pytest.fixture
def mock_main_repository():
    """メインリポジトリのモックを作成します"""
    repo = AsyncMock(spec=WellnessRepositoryInterface)
    return repo

@pytest.fixture
def redis_wellness_repository(mock_redis_service, mock_main_repository):
    """テスト用のRedisWellnessRepositoryインスタンスを作成します"""
    return RedisWellnessRepository(
        redis_service=mock_redis_service,
        main_repository=mock_main_repository,
        ttl_seconds=3600
    )

@pytest.fixture
def sample_wellness_score():
    """テスト用のWellnessScoreオブジェクトを作成します"""
    return WellnessScore(
        id="test-score-id",
        company_id="test-company-id",
        total_score=75.5,
        category_scores={
            ScoreCategory.FINANCIAL: 80.0,
            ScoreCategory.HEALTH: 70.0,
            ScoreCategory.WORK_LIFE_BALANCE: 75.0
        },
        metrics=[
            ScoreMetric(
                name="収益率",
                value=85.0,
                weight=1.0,
                category=ScoreCategory.FINANCIAL,
                timestamp=datetime.now(),
                raw_data={"source": "financial_data"}
            )
        ],
        timestamp=datetime.now(),
        created_by="test-user-id"
    )

@pytest.fixture
def sample_recommendation_plan():
    """テスト用のRecommendationPlanオブジェクトを作成します"""
    return RecommendationPlan(
        company_id="test-company-id",
        score_id="test-score-id",
        actions=[
            RecommendationAction(
                id="test-action-id",
                title="コスト削減",
                description="不要な経費を削減する",
                category=ScoreCategory.FINANCIAL,
                impact_level=4,
                effort_level=2,
                time_frame="short",
                resources=["経理部門", "各部門マネージャー"]
            )
        ],
        generated_at=datetime.now(),
        generated_by="test-user-id"
    )

@pytest.mark.asyncio
async def test_save_score(redis_wellness_repository, mock_redis_service, mock_main_repository, sample_wellness_score):
    """スコア保存のテスト"""
    # 準備
    mock_main_repository.save_score.return_value = sample_wellness_score

    # 実行
    result = await redis_wellness_repository.save_score(sample_wellness_score)

    # 検証
    mock_main_repository.save_score.assert_called_once_with(sample_wellness_score)
    assert result == sample_wellness_score

    # キャッシュが更新されたことを確認
    mock_redis_service.set_json.assert_any_call(
        f"wellness:score:{sample_wellness_score.id}",
        redis_wellness_repository._serialize_score(sample_wellness_score),
        3600
    )
    mock_redis_service.set_json.assert_any_call(
        f"wellness:company:latest:{sample_wellness_score.company_id}",
        redis_wellness_repository._serialize_score(sample_wellness_score),
        3600
    )

    # 関連キャッシュが無効化されたことを確認
    mock_redis_service.delete_key.assert_any_call(
        f"wellness:company:scores:{sample_wellness_score.company_id}"
    )
    mock_redis_service.delete_key.assert_any_call(
        f"wellness:company:history:{sample_wellness_score.company_id}"
    )

@pytest.mark.asyncio
async def test_get_score_by_id_cached(redis_wellness_repository, mock_redis_service, sample_wellness_score):
    """キャッシュからスコアをIDで取得するテスト"""
    # 準備
    score_id = sample_wellness_score.id
    score_data = redis_wellness_repository._serialize_score(sample_wellness_score)

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = score_data

    # 実行
    result = await redis_wellness_repository.get_score_by_id(score_id)

    # 検証
    mock_redis_service.get_json.assert_called_once_with(f"wellness:score:{score_id}")
    assert isinstance(result, WellnessScore)
    assert result.id == score_id
    assert result.company_id == sample_wellness_score.company_id
    assert result.total_score == sample_wellness_score.total_score

@pytest.mark.asyncio
async def test_get_score_by_id_not_cached(redis_wellness_repository, mock_redis_service, mock_main_repository, sample_wellness_score):
    """キャッシュにないスコアをメインリポジトリから取得するテスト"""
    # 準備
    score_id = sample_wellness_score.id

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = None

    # メインリポジトリからの応答をモック
    mock_main_repository.get_score_by_id.return_value = sample_wellness_score

    # 実行
    result = await redis_wellness_repository.get_score_by_id(score_id)

    # 検証
    mock_redis_service.get_json.assert_called_once_with(f"wellness:score:{score_id}")
    mock_main_repository.get_score_by_id.assert_called_once_with(score_id)
    assert result == sample_wellness_score

    # キャッシュが更新されたことを確認
    mock_redis_service.set_json.assert_called_once_with(
        f"wellness:score:{score_id}",
        redis_wellness_repository._serialize_score(sample_wellness_score),
        3600
    )

@pytest.mark.asyncio
async def test_get_latest_score_cached(redis_wellness_repository, mock_redis_service, sample_wellness_score):
    """キャッシュから最新スコアを取得するテスト"""
    # 準備
    company_id = sample_wellness_score.company_id
    score_data = redis_wellness_repository._serialize_score(sample_wellness_score)

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = score_data

    # 実行
    result = await redis_wellness_repository.get_latest_score(company_id)

    # 検証
    mock_redis_service.get_json.assert_called_once_with(f"wellness:company:latest:{company_id}")
    assert isinstance(result, WellnessScore)
    assert result.id == sample_wellness_score.id
    assert result.company_id == company_id

@pytest.mark.asyncio
async def test_get_latest_score_not_cached(redis_wellness_repository, mock_redis_service, mock_main_repository, sample_wellness_score):
    """キャッシュにない最新スコアをメインリポジトリから取得するテスト"""
    # 準備
    company_id = sample_wellness_score.company_id

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = None

    # メインリポジトリからの応答をモック
    mock_main_repository.get_latest_score.return_value = sample_wellness_score

    # 実行
    result = await redis_wellness_repository.get_latest_score(company_id)

    # 検証
    mock_redis_service.get_json.assert_called_once_with(f"wellness:company:latest:{company_id}")
    mock_main_repository.get_latest_score.assert_called_once_with(company_id)
    assert result == sample_wellness_score

    # キャッシュが更新されたことを確認
    mock_redis_service.set_json.assert_called_once_with(
        f"wellness:company:latest:{company_id}",
        redis_wellness_repository._serialize_score(sample_wellness_score),
        3600
    )

@pytest.mark.asyncio
async def test_get_scores_by_company_cached(redis_wellness_repository, mock_redis_service, sample_wellness_score):
    """キャッシュから会社のスコア一覧を取得するテスト"""
    # 準備
    company_id = sample_wellness_score.company_id
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    limit = 10

    # キャッシュキーを生成
    cache_params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "limit": limit
    }
    cache_key = f"wellness:company:scores:{company_id}:{json.dumps(cache_params)}"

    # スコア一覧データ
    scores_data = [redis_wellness_repository._serialize_score(sample_wellness_score)]

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = scores_data

    # 実行
    result = await redis_wellness_repository.get_scores_by_company(
        company_id, start_date, end_date, limit
    )

    # 検証
    mock_redis_service.get_json.assert_called_once()
    assert len(result) == 1
    assert isinstance(result[0], WellnessScore)
    assert result[0].id == sample_wellness_score.id

@pytest.mark.asyncio
async def test_get_scores_by_company_not_cached(redis_wellness_repository, mock_redis_service, mock_main_repository, sample_wellness_score):
    """キャッシュにない会社のスコア一覧をメインリポジトリから取得するテスト"""
    # 準備
    company_id = sample_wellness_score.company_id
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    limit = 10

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = None

    # メインリポジトリからの応答をモック
    mock_main_repository.get_scores_by_company.return_value = [sample_wellness_score]

    # 実行
    result = await redis_wellness_repository.get_scores_by_company(
        company_id, start_date, end_date, limit
    )

    # 検証
    mock_redis_service.get_json.assert_called_once()
    mock_main_repository.get_scores_by_company.assert_called_once_with(
        company_id, start_date, end_date, limit
    )
    assert len(result) == 1
    assert result[0] == sample_wellness_score

    # キャッシュが更新されたことを確認
    mock_redis_service.set_json.assert_called_once()

@pytest.mark.asyncio
async def test_save_recommendation_plan(redis_wellness_repository, mock_redis_service, mock_main_repository, sample_recommendation_plan):
    """推奨プラン保存のテスト"""
    # 準備
    mock_main_repository.save_recommendation_plan.return_value = sample_recommendation_plan

    # 実行
    result = await redis_wellness_repository.save_recommendation_plan(sample_recommendation_plan)

    # 検証
    mock_main_repository.save_recommendation_plan.assert_called_once_with(sample_recommendation_plan)
    assert result == sample_recommendation_plan

    # キャッシュが更新されたことを確認（スコアIDなしのキー）
    mock_redis_service.set_json.assert_any_call(
        f"wellness:plan:{sample_recommendation_plan.company_id}",
        redis_wellness_repository._serialize_recommendation_plan(sample_recommendation_plan),
        3600
    )

    # キャッシュが更新されたことを確認（スコアIDありのキー）
    mock_redis_service.set_json.assert_any_call(
        f"wellness:plan:{sample_recommendation_plan.company_id}:{sample_recommendation_plan.score_id}",
        redis_wellness_repository._serialize_recommendation_plan(sample_recommendation_plan),
        3600
    )

@pytest.mark.asyncio
async def test_get_recommendation_plan_cached(redis_wellness_repository, mock_redis_service, sample_recommendation_plan):
    """キャッシュから推奨プランを取得するテスト"""
    # 準備
    company_id = sample_recommendation_plan.company_id
    score_id = sample_recommendation_plan.score_id
    plan_data = redis_wellness_repository._serialize_recommendation_plan(sample_recommendation_plan)

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = plan_data

    # 実行
    result = await redis_wellness_repository.get_recommendation_plan(company_id, score_id)

    # 検証
    mock_redis_service.get_json.assert_called_once_with(f"wellness:plan:{company_id}:{score_id}")
    assert isinstance(result, RecommendationPlan)
    assert result.company_id == company_id
    assert result.score_id == score_id
    assert len(result.actions) == len(sample_recommendation_plan.actions)

@pytest.mark.asyncio
async def test_get_recommendation_plan_not_cached(redis_wellness_repository, mock_redis_service, mock_main_repository, sample_recommendation_plan):
    """キャッシュにない推奨プランをメインリポジトリから取得するテスト"""
    # 準備
    company_id = sample_recommendation_plan.company_id
    score_id = sample_recommendation_plan.score_id

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = None

    # メインリポジトリからの応答をモック
    mock_main_repository.get_recommendation_plan.return_value = sample_recommendation_plan

    # 実行
    result = await redis_wellness_repository.get_recommendation_plan(company_id, score_id)

    # 検証
    mock_redis_service.get_json.assert_called_once_with(f"wellness:plan:{company_id}:{score_id}")
    mock_main_repository.get_recommendation_plan.assert_called_once_with(company_id, score_id)
    assert result == sample_recommendation_plan

    # キャッシュが更新されたことを確認
    mock_redis_service.set_json.assert_called_once_with(
        f"wellness:plan:{company_id}:{score_id}",
        redis_wellness_repository._serialize_recommendation_plan(sample_recommendation_plan),
        3600
    )

@pytest.mark.asyncio
async def test_get_company_benchmarks_cached(redis_wellness_repository, mock_redis_service):
    """キャッシュから会社ベンチマークを取得するテスト"""
    # 準備
    company_id = "test-company-id"
    category = ScoreCategory.FINANCIAL
    benchmark_data = {
        "company": 75.5,
        "industry_avg": 68.2,
        "industry_top": 92.1,
        "all_avg": 65.9
    }

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = benchmark_data

    # 実行
    result = await redis_wellness_repository.get_company_benchmarks(company_id, category)

    # 検証
    mock_redis_service.get_json.assert_called_once_with(f"wellness:benchmark:{company_id}:{category.value}")
    assert result == benchmark_data

@pytest.mark.asyncio
async def test_get_company_benchmarks_not_cached(redis_wellness_repository, mock_redis_service, mock_main_repository):
    """キャッシュにない会社ベンチマークをメインリポジトリから取得するテスト"""
    # 準備
    company_id = "test-company-id"
    category = ScoreCategory.FINANCIAL
    benchmark_data = {
        "company": 75.5,
        "industry_avg": 68.2,
        "industry_top": 92.1,
        "all_avg": 65.9
    }

    # Redisからの応答をモック
    mock_redis_service.get_json.return_value = None

    # メインリポジトリからの応答をモック
    mock_main_repository.get_company_benchmarks.return_value = benchmark_data

    # 実行
    result = await redis_wellness_repository.get_company_benchmarks(company_id, category)

    # 検証
    mock_redis_service.get_json.assert_called_once_with(f"wellness:benchmark:{company_id}:{category.value}")
    mock_main_repository.get_company_benchmarks.assert_called_once_with(company_id, category)
    assert result == benchmark_data

    # キャッシュが更新されたことを確認
    mock_redis_service.set_json.assert_called_once_with(
        f"wellness:benchmark:{company_id}:{category.value}",
        benchmark_data,
        3600
    )

@pytest.mark.asyncio
async def test_serialization_deserialization(redis_wellness_repository, sample_wellness_score):
    """シリアライズとデシリアライズのテスト"""
    # シリアライズ
    serialized = redis_wellness_repository._serialize_score(sample_wellness_score)

    # 検証
    assert serialized["id"] == sample_wellness_score.id
    assert serialized["company_id"] == sample_wellness_score.company_id
    assert serialized["total_score"] == sample_wellness_score.total_score
    assert "category_scores" in serialized
    assert "metrics" in serialized

    # デシリアライズ
    deserialized = redis_wellness_repository._deserialize_score(serialized)

    # 検証
    assert isinstance(deserialized, WellnessScore)
    assert deserialized.id == sample_wellness_score.id
    assert deserialized.company_id == sample_wellness_score.company_id
    assert deserialized.total_score == sample_wellness_score.total_score
    assert len(deserialized.category_scores) == len(sample_wellness_score.category_scores)
    assert len(deserialized.metrics) == len(sample_wellness_score.metrics)