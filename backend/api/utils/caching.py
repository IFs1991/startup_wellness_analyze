"""
可視化キャッシュユーティリティ

このモジュールでは、可視化リクエストとそのレスポンスをキャッシュするメカニズムを提供します。
高頻度な同一リクエストに対するパフォーマンスを向上させるためのメモリキャッシュを実装します。
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import functools
import asyncio

logger = logging.getLogger(__name__)

# メモリ内キャッシュ
_cache: Dict[str, Dict[str, Any]] = {}
# キャッシュ項目の期限情報
_cache_expiry: Dict[str, datetime] = {}
# キャッシュサイズの上限（項目数）
MAX_CACHE_SIZE = 1000
# デフォルトの有効期限（秒）
DEFAULT_EXPIRY_SECONDS = 3600


def _generate_cache_key(analysis_type: str, analysis_results: Dict[str, Any],
                       visualization_type: str, options: Dict[str, Any]) -> str:
    """
    可視化リクエストからキャッシュキーを生成します。

    Args:
        analysis_type: 分析タイプ
        analysis_results: 分析結果
        visualization_type: 可視化タイプ
        options: 可視化オプション

    Returns:
        キャッシュキー
    """
    # キーの生成に使用するデータを整理
    key_data = {
        "analysis_type": analysis_type,
        # 分析結果は大きい可能性があるため、サマリーのみ使用
        "analysis_summary": _get_analysis_summary(analysis_results),
        "visualization_type": visualization_type,
        "options": options
    }

    # JSONに変換してハッシュを計算
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_json.encode('utf-8')).hexdigest()


def _get_analysis_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    大きな分析結果からサマリー情報を抽出します。

    Args:
        analysis_results: 分析結果

    Returns:
        サマリー情報
    """
    # 結果が特定のキーを持つ場合の特別処理
    if "metadata" in analysis_results:
        return analysis_results["metadata"]

    # データ量が多いキーは除外
    summary = {}
    for key, value in analysis_results.items():
        if key in ["data", "raw_data", "detailed_results"]:
            # データ量が多いキーはスキップ
            continue
        if isinstance(value, (dict, list)) and len(str(value)) > 1000:
            # 大きなオブジェクトはスキップ
            continue
        summary[key] = value

    return summary


def get_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    キャッシュからデータを取得します。

    Args:
        cache_key: キャッシュキー

    Returns:
        キャッシュされているデータ、または None
    """
    now = datetime.now()

    # キャッシュの有効期限をチェック
    if cache_key in _cache_expiry and now > _cache_expiry[cache_key]:
        # 期限切れの場合、キャッシュから削除
        logger.debug(f"キャッシュ期限切れ: {cache_key}")
        del _cache[cache_key]
        del _cache_expiry[cache_key]
        return None

    return _cache.get(cache_key)


def add_to_cache(cache_key: str, data: Dict[str, Any], expiry_seconds: int = DEFAULT_EXPIRY_SECONDS) -> None:
    """
    データをキャッシュに追加します。

    Args:
        cache_key: キャッシュキー
        data: キャッシュするデータ
        expiry_seconds: キャッシュの有効期間（秒）
    """
    # キャッシュが最大サイズに達した場合、最も古いエントリを削除
    if len(_cache) >= MAX_CACHE_SIZE:
        oldest_key = min(_cache_expiry, key=_cache_expiry.get)
        del _cache[oldest_key]
        del _cache_expiry[oldest_key]
        logger.debug(f"キャッシュ最大サイズに達したため最も古いエントリを削除: {oldest_key}")

    # キャッシュに追加
    _cache[cache_key] = data
    _cache_expiry[cache_key] = datetime.now() + timedelta(seconds=expiry_seconds)
    logger.debug(f"キャッシュに追加: {cache_key}, 有効期限: {expiry_seconds}秒")


def clear_cache() -> None:
    """
    キャッシュ全体をクリアします。
    """
    _cache.clear()
    _cache_expiry.clear()
    logger.info("キャッシュをクリアしました")


def clear_expired_cache() -> int:
    """
    期限切れのキャッシュエントリをクリアします。

    Returns:
        削除されたエントリの数
    """
    now = datetime.now()
    expired_keys = [k for k, v in _cache_expiry.items() if now > v]

    for key in expired_keys:
        del _cache[key]
        del _cache_expiry[key]

    logger.info(f"{len(expired_keys)}個の期限切れキャッシュエントリを削除しました")
    return len(expired_keys)


def async_cache(expiry_seconds: int = DEFAULT_EXPIRY_SECONDS):
    """
    非同期関数の結果をキャッシュするデコレータ。

    Args:
        expiry_seconds: キャッシュの有効期間（秒）

    Returns:
        デコレータ関数
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # キャッシュキーの生成
            cache_input = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            key_json = json.dumps(cache_input, sort_keys=True, default=str)
            cache_key = hashlib.md5(key_json.encode('utf-8')).hexdigest()

            # キャッシュから取得
            cached_result = get_from_cache(cache_key)
            if cached_result is not None:
                logger.debug(f"キャッシュヒット: {func.__name__}")
                return cached_result

            # キャッシュにない場合は関数を実行
            logger.debug(f"キャッシュミス: {func.__name__}")
            result = await func(*args, **kwargs)

            # 結果をキャッシュに保存
            add_to_cache(cache_key, result, expiry_seconds)

            return result
        return wrapper
    return decorator


# 定期的にキャッシュクリーンアップを実行するタスク
async def periodic_cache_cleanup(interval_seconds: int = 600):
    """
    定期的に期限切れのキャッシュをクリーンアップするタスク。

    Args:
        interval_seconds: クリーンアップの間隔（秒）
    """
    while True:
        await asyncio.sleep(interval_seconds)
        cleared_count = clear_expired_cache()
        logger.info(f"定期キャッシュクリーンアップ完了: {cleared_count}個のエントリを削除")