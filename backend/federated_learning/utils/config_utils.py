"""
設定ユーティリティ

このモジュールは、連合学習システムの設定を処理するユーティリティ関数を提供します。
設定ファイルの読み込みや検証などの機能が含まれます。
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """設定ファイルを読み込む

    Args:
        config_path: 設定ファイルパス（省略時はデフォルト設定を使用）

    Returns:
        設定辞書

    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
        yaml.YAMLError: YAML解析エラーが発生した場合
    """
    if config_path is None:
        # デフォルトの設定ファイルパス
        config_path = Path(__file__).parents[1] / "config.yaml"

    logger.info(f"設定ファイルを読み込んでいます: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.info("設定ファイルを正常に読み込みました")
        return config

    except FileNotFoundError:
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        raise

    except yaml.YAMLError as e:
        logger.error(f"設定ファイルの解析エラー: {e}")
        raise

def validate_config(config: Dict[str, Any]) -> List[str]:
    """設定を検証する

    Args:
        config: 検証する設定辞書

    Returns:
        エラーメッセージのリスト（エラーがない場合は空リスト）
    """
    errors = []

    # 必須セクションの確認
    required_sections = ["system", "federated_learning"]
    for section in required_sections:
        if section not in config:
            errors.append(f"必須セクション '{section}' が見つかりません")

    # 連合学習セクションの検証（存在する場合）
    if "federated_learning" in config:
        fl_config = config["federated_learning"]

        # モデル設定の検証
        if "models" not in fl_config:
            errors.append("連合学習セクションに 'models' が見つかりません")
        elif not isinstance(fl_config["models"], list):
            errors.append("'models' はリストである必要があります")
        else:
            for i, model in enumerate(fl_config["models"]):
                # モデル名の確認
                if "name" not in model:
                    errors.append(f"モデル {i} に 'name' が見つかりません")

                # アーキテクチャの確認
                if "architecture" not in model:
                    errors.append(f"モデル '{model.get('name', f'{i}')}' に 'architecture' が見つかりません")

                # 訓練設定の確認
                if "training" not in model:
                    errors.append(f"モデル '{model.get('name', f'{i}')}' に 'training' が見つかりません")

                # 連合設定の確認
                if "federated_settings" not in model:
                    errors.append(f"モデル '{model.get('name', f'{i}')}' に 'federated_settings' が見つかりません")

        # クライアント設定の検証
        if "client" not in fl_config:
            errors.append("連合学習セクションに 'client' が見つかりません")

        # サーバー設定の検証
        if "central_server" not in fl_config:
            errors.append("連合学習セクションに 'central_server' が見つかりません")

        # セキュリティ設定の検証
        security_sections = ["differential_privacy", "secure_aggregation"]
        for section in security_sections:
            if section not in fl_config:
                errors.append(f"連合学習セクションに '{section}' が見つかりません")

    if errors:
        logger.warning(f"設定の検証で {len(errors)} 個のエラーが見つかりました")
        for error in errors:
            logger.warning(f"設定エラー: {error}")
    else:
        logger.info("設定の検証が成功しました")

    return errors

def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """特定のモデルの設定を取得する

    Args:
        config: 設定辞書
        model_name: モデル名

    Returns:
        モデル設定

    Raises:
        ValueError: 指定されたモデルが見つからない場合
    """
    if "federated_learning" not in config or "models" not in config["federated_learning"]:
        raise ValueError("連合学習またはモデル設定が見つかりません")

    for model in config["federated_learning"]["models"]:
        if model.get("name") == model_name:
            return model

    raise ValueError(f"モデル '{model_name}' が設定に見つかりません")