from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session
from database.database import get_db
from typing import List, Dict, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 基本的な機械学習モデルのインポート
from sklearn.ensemble import RandomForestRegressor

# Performance Predictor クラスの実装
class PerformancePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False

    def predict(self, data: pd.DataFrame, target_variable: str) -> pd.DataFrame:
        """
        パフォーマンスを予測する
        """
        if not self.is_trained:
            raise HTTPException(status_code=400, detail="Model needs to be trained first")

        try:
            features = data.drop(columns=[target_variable], errors='ignore')
            predictions = self.model.predict(features)
            result_df = pd.DataFrame({'predicted_value': predictions})
            return result_df
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    def train(self, data: pd.DataFrame, target_variable: str):
        """
        モデルを訓練する
        """
        try:
            X = data.drop(columns=[target_variable])
            y = data[target_variable]
            self.model.fit(X, y)
            self.is_trained = True
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Training error: {str(e)}")

# Model Evaluator クラスの実装
class ModelEvaluator:
    def evaluate(self, data: pd.DataFrame, target_variable: str) -> Dict:
        """
        モデルのパフォーマンスを評価する
        """
        try:
            X = data.drop(columns=[target_variable])
            y = data[target_variable]

            # データを訓練セットとテストセットに分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # モデルのインスタンス化と訓練
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # 予測と評価
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            return {
                "mse": float(mse),
                "r2": float(r2),
                "train_size": len(X_train),
                "test_size": len(X_test)
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Evaluation error: {str(e)}")

# 入力データのモデルを定義
class PredictionInput(BaseModel):
    data: List[Dict]
    target_variable: str

    class Config:
        schema_extra = {
            "example": {
                "data": [{"feature1": 1.0, "feature2": 2.0, "target": 3.0}],
                "target_variable": "target"
            }
        }

# ルーターの設定
router = APIRouter(
    prefix="/prediction",
    tags=["prediction"],
    responses={404: {"description": "Not found"}}
)

@router.post("/performance/")
async def predict_performance(
    input_data: PredictionInput = Body(...),
    db: Session = Depends(get_db)
):
    """将来パフォーマンスを予測します。"""
    try:
        predictor = PerformancePredictor()
        df = pd.DataFrame(input_data.data)

        # まずモデルを訓練
        predictor.train(df, input_data.target_variable)

        # 予測を実行
        predictions = predictor.predict(
            data=df,
            target_variable=input_data.target_variable
        )
        return predictions.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/model_evaluation/")
async def evaluate_model_performance(
    input_data: PredictionInput = Body(...),
    db: Session = Depends(get_db)
):
    """モデルパフォーマンスを評価します。"""
    try:
        evaluator = ModelEvaluator()
        df = pd.DataFrame(input_data.data)
        evaluation = evaluator.evaluate(
            data=df,
            target_variable=input_data.target_variable
        )
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/retrain_model/")
async def retrain_model(
    input_data: PredictionInput = Body(...),
    db: Session = Depends(get_db)
):
    """モデルを再訓練します。"""
    try:
        predictor = PerformancePredictor()
        df = pd.DataFrame(input_data.data)
        predictor.train(df, input_data.target_variable)
        return {"message": "Model retraining completed successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))