"""
共通ロギングモジュール
アプリケーション全体で一貫したロギング設定を提供します。
"""

import logging
from typing import Optional

def get_logger(module_name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    指定されたモジュール名で設定済みのロガーを取得します。

    Args:
        module_name: ロガーの名前（通常はモジュール名）
        log_level: ロギングレベル（デフォルト: INFO）

    Returns:
        設定済みのLoggerインスタンス
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    # ハンドラが未設定の場合のみ設定
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def setup_application_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> None:
    """
    アプリケーション全体のロギング設定を行います。

    Args:
        log_level: ロギングレベル（デフォルト: INFO）
        log_file: ログファイルのパス（指定された場合はファイル出力も追加）
    """
    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 既存のハンドラをクリア
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # コンソール出力の設定
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # ファイル出力の設定（指定された場合）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # サードパーティライブラリのロギングレベルを調整
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)