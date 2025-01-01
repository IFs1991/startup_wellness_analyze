from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, ValidationError
import pandas as pd
import numpy as np
from enum import Enum

class DataValidationRule(BaseModel):
    field: str
    rule_type: str
    parameters: Dict[str, Any] = {}
    error_message: str

class DataValidationResult(BaseModel):
    is_valid: bool
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    validated_at: datetime

class DataTransformation(BaseModel):
    field: str
    transformation_type: str
    parameters: Dict[str, Any] = {}

class ProcessingResult(BaseModel):
    data_id: str
    original_data: Dict[str, Any]
    processed_data: Dict[str, Any]
    validation_result: DataValidationResult
    transformations_applied: List[str]
    processed_at: datetime

class DataProcessor:
    def __init__(self, database):
        self.db = database

    async def validate_data(
        self,
        data: Dict[str, Any],
        validation_rules: List[DataValidationRule]
    ) -> DataValidationResult:
        """データを検証する"""
        errors = []
        warnings = []

        for rule in validation_rules:
            try:
                is_valid = await self._apply_validation_rule(data, rule)
                if not is_valid:
                    errors.append({
                        "field": rule.field,
                        "message": rule.error_message
                    })
            except Exception as e:
                warnings.append({
                    "field": rule.field,
                    "message": str(e)
                })

        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validated_at=datetime.utcnow()
        )

    async def clean_data(
        self,
        data: Dict[str, Any],
        cleaning_rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """データをクリーニングする"""
        cleaned_data = data.copy()

        for rule in cleaning_rules:
            field = rule.get("field")
            method = rule.get("method")
            params = rule.get("parameters", {})

            if field in cleaned_data:
                cleaned_data[field] = await self._apply_cleaning_method(
                    cleaned_data[field],
                    method,
                    params
                )

        return cleaned_data

    async def transform_data(
        self,
        data: Dict[str, Any],
        transformations: List[DataTransformation]
    ) -> Dict[str, Any]:
        """データを変換する"""
        transformed_data = data.copy()

        for transform in transformations:
            if transform.field in transformed_data:
                transformed_data[transform.field] = await self._apply_transformation(
                    transformed_data[transform.field],
                    transform.transformation_type,
                    transform.parameters
                )

        return transformed_data

    async def process_data(
        self,
        data: Dict[str, Any],
        validation_rules: List[DataValidationRule],
        cleaning_rules: List[Dict[str, Any]],
        transformations: List[DataTransformation]
    ) -> ProcessingResult:
        """データを処理する"""
        # バリデーション
        validation_result = await self.validate_data(data, validation_rules)
        if not validation_result.is_valid:
            return ProcessingResult(
                data_id=self._generate_id(),
                original_data=data,
                processed_data=data,
                validation_result=validation_result,
                transformations_applied=[],
                processed_at=datetime.utcnow()
            )

        # クリーニング
        cleaned_data = await self.clean_data(data, cleaning_rules)

        # 変換
        transformed_data = await self.transform_data(cleaned_data, transformations)

        return ProcessingResult(
            data_id=self._generate_id(),
            original_data=data,
            processed_data=transformed_data,
            validation_result=validation_result,
            transformations_applied=[t.transformation_type for t in transformations],
            processed_at=datetime.utcnow()
        )

    async def _apply_validation_rule(
        self,
        data: Dict[str, Any],
        rule: DataValidationRule
    ) -> bool:
        """バリデーションルールを適用する"""
        value = data.get(rule.field)

        validation_methods = {
            "required": lambda v, p: v is not None,
            "type": lambda v, p: isinstance(v, p.get("type")),
            "range": lambda v, p: p.get("min", float("-inf")) <= v <= p.get("max", float("inf")),
            "pattern": lambda v, p: bool(re.match(p.get("pattern"), str(v))),
            "enum": lambda v, p: v in p.get("values", []),
            "custom": lambda v, p: self._custom_validation(v, p)
        }

        validator = validation_methods.get(rule.rule_type)
        if not validator:
            raise ValueError(f"Unknown validation rule type: {rule.rule_type}")

        return validator(value, rule.parameters)

    async def _apply_cleaning_method(
        self,
        value: Any,
        method: str,
        params: Dict[str, Any]
    ) -> Any:
        """クリーニングメソッドを適用する"""
        cleaning_methods = {
            "remove_whitespace": lambda v, p: str(v).strip(),
            "replace_null": lambda v, p: p.get("default") if v is None else v,
            "remove_special_chars": lambda v, p: re.sub(r"[^\w\s]", "", str(v)),
            "normalize_case": lambda v, p: str(v).lower() if p.get("case") == "lower" else str(v).upper(),
            "custom": lambda v, p: self._custom_cleaning(v, p)
        }

        cleaner = cleaning_methods.get(method)
        if not cleaner:
            raise ValueError(f"Unknown cleaning method: {method}")

        return cleaner(value, params)

    async def _apply_transformation(
        self,
        value: Any,
        transformation_type: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """変換を適用する"""
        transformation_methods = {
            "format_date": lambda v, p: self._format_date(v, p),
            "convert_type": lambda v, p: self._convert_type(v, p),
            "scale_numeric": lambda v, p: self._scale_numeric(v, p),
            "encode_categorical": lambda v, p: self._encode_categorical(v, p),
            "custom": lambda v, p: self._custom_transformation(v, p)
        }

        transformer = transformation_methods.get(transformation_type)
        if not transformer:
            raise ValueError(f"Unknown transformation type: {transformation_type}")

        return transformer(value, parameters)

    def _format_date(self, value: Any, params: Dict[str, Any]) -> str:
        """日付をフォーマットする"""
        if isinstance(value, str):
            dt = datetime.strptime(value, params.get("input_format", "%Y-%m-%d"))
        elif isinstance(value, datetime):
            dt = value
        else:
            raise ValueError("Invalid date value")

        return dt.strftime(params.get("output_format", "%Y-%m-%d"))

    def _convert_type(self, value: Any, params: Dict[str, Any]) -> Any:
        """型を変換する"""
        target_type = params.get("target_type")
        type_converters = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool
        }

        converter = type_converters.get(target_type)
        if not converter:
            raise ValueError(f"Unsupported target type: {target_type}")

        return converter(value)

    def _scale_numeric(self, value: float, params: Dict[str, Any]) -> float:
        """数値をスケーリングする"""
        method = params.get("method", "minmax")
        if method == "minmax":
            min_val = params.get("min", 0)
            max_val = params.get("max", 1)
            return (value - min_val) / (max_val - min_val)
        elif method == "standard":
            mean = params.get("mean", 0)
            std = params.get("std", 1)
            return (value - mean) / std
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

    def _encode_categorical(self, value: str, params: Dict[str, Any]) -> Any:
        """カテゴリカル変数をエンコードする"""
        method = params.get("method", "label")
        if method == "label":
            mapping = params.get("mapping", {})
            return mapping.get(value, value)
        elif method == "onehot":
            categories = params.get("categories", [])
            return [1 if value == cat else 0 for cat in categories]
        else:
            raise ValueError(f"Unsupported encoding method: {method}")

    def _custom_validation(self, value: Any, params: Dict[str, Any]) -> bool:
        """カスタムバリデーションを実行する"""
        # TODO: カスタムバリデーションロジックを実装
        return True

    def _custom_cleaning(self, value: Any, params: Dict[str, Any]) -> Any:
        """カスタムクリーニングを実行する"""
        # TODO: カスタムクリーニングロジックを実装
        return value

    def _custom_transformation(self, value: Any, params: Dict[str, Any]) -> Any:
        """カスタム変換を実行する"""
        # TODO: カスタム変換ロジックを実装
        return value

    def _generate_id(self) -> str:
        """IDを生成する"""
        import uuid
        return str(uuid.uuid4())