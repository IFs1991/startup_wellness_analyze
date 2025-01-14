"""
CSVデータ処理のユーティリティ関数を提供します。
Firestoreとの連携を考慮した実装になっています。
"""
from io import StringIO
from typing import Dict, List, Union, Optional
import pandas as pd
from fastapi import HTTPException
import logging
from datetime import datetime

# ロギングの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DataProcessingError(Exception):
    """データ処理に関するエラー"""
    pass

async def convert_csv_to_dataframe(
    csv_data: str,
    date_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    文字列型のCSVデータをpandasのDataFrameに変換し、
    Firestore互換のデータ型に調整します。

    Args:
        csv_data (str): CSVデータ
        date_columns (Optional[List[str]]): 日付型として処理するカラム名のリスト
        numeric_columns (Optional[List[str]]): 数値型として処理するカラム名のリスト

    Returns:
        pd.DataFrame: 変換後のDataFrame

    Raises:
        HTTPException: CSV変換時のエラー
        DataProcessingError: データ処理時のエラー
    """
    try:
        logger.info("Starting CSV to DataFrame conversion")

        # CSVデータをDataFrameに変換
        df = pd.read_csv(StringIO(csv_data))

        # 日付カラムの処理
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col]).apply(
                        lambda x: x.isoformat() if pd.notnull(x) else None
                    )

        # 数値カラムの処理
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # NaN値をNoneに変換（Firestore互換）
        df = df.replace({pd.NA: None, pd.NaT: None})

        logger.info(f"Successfully converted CSV data to DataFrame with {len(df)} rows")
        return df

    except pd.errors.ParserError as e:
        error_msg = f"Invalid CSV data format: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    except Exception as e:
        error_msg = f"Error processing CSV data: {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg)

async def prepare_firestore_data(
    df: pd.DataFrame,
    metadata: Optional[Dict] = None
) -> List[Dict]:
    """
    DataFrameをFirestore互換の形式に変換します。

    Args:
        df (pd.DataFrame): 変換対象のDataFrame
        metadata (Optional[Dict]): 追加のメタデータ

    Returns:
        List[Dict]: Firestore互換のデータリスト

    Raises:
        DataProcessingError: データ変換時のエラー
    """
    try:
        logger.info("Starting DataFrame to Firestore format conversion")

        # DataFrameを辞書のリストに変換
        records = df.to_dict('records')

        # Firestore互換のデータ構造に変換
        firestore_data = []
        for record in records:
            # メタデータの追加
            document = {
                'data': record,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }

            # オプションのメタデータを追加
            if metadata:
                document['metadata'] = metadata

            firestore_data.append(document)

        logger.info(f"Successfully converted {len(firestore_data)} records to Firestore format")
        return firestore_data

    except Exception as e:
        error_msg = f"Error preparing data for Firestore: {str(e)}"
        logger.error(error_msg)
        raise DataProcessingError(error_msg)

async def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    column_types: Optional[Dict[str, str]] = None
) -> bool:
    """
    DataFrameのスキーマを検証します。

    Args:
        df (pd.DataFrame): 検証対象のDataFrame
        required_columns (List[str]): 必須カラムのリスト
        column_types (Optional[Dict[str, str]]): カラム名と期待される型の辞書

    Returns:
        bool: 検証結果

    Raises:
        HTTPException: スキーマ検証エラー
    """
    try:
        logger.info("Starting DataFrame schema validation")

        # 必須カラムの存在チェック
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )

        # カラムの型チェック
        if column_types:
            for col, expected_type in column_types.items():
                if col in df.columns:
                    actual_type = df[col].dtype.name
                    if actual_type != expected_type:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Column {col} has type {actual_type}, expected {expected_type}"
                        )

        logger.info("DataFrame schema validation successful")
        return True

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error validating DataFrame schema: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)