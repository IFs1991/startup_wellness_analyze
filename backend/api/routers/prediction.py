from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session
from database.database import get_db
from core.performance_predictor import PerformancePredictor
from core.model_evaluator import ModelEvaluator
import pandas as pd
from typing import List, Dict
from pydantic import BaseModel

# ルーターの設定
router = APIRouter(
    prefix="/prediction",
    tags=["prediction"],
    responses={404: {"description": "Not found"}},
)

# 入力データのモデルを定義
class PredictionInput(BaseModel):
    data: List[Dict]  # DataFrameに変換可能なデータ形式
    target_variable: str

@router.post("/performance/")
async def predict_performance(
    input_data: PredictionInput = Body(...),  # Body(...)はリクエストボディからデータを取得することを示す
    db: Session = Depends(get_db)
):
    """将来パフォーマンスを予測します。"""
    predictor = PerformancePredictor()

    # 入力データをDataFrameに変換
    df = pd.DataFrame(input_data.data)

    # 予測を実行
    predictions = predictor.predict(
        data=df,
        target_variable=input_data.target_variable
    )

    # DataFrameをJSONに変換して返す
    return predictions.to_dict(orient='records')

@router.post("/model_evaluation/")  # GETからPOSTに変更
async def evaluate_model_performance(
    input_data: PredictionInput = Body(...),
    db: Session = Depends(get_db)
):
    """モデルパフォーマンスを評価します。"""
    evaluator = ModelEvaluator()

    # 入力データをDataFrameに変換
    df = pd.DataFrame(input_data.data)

    # 評価を実行
    evaluation = evaluator.evaluate(
        data=df,
        target_variable=input_data.target_variable
    )

    # 評価結果を返す
    return evaluation

# モデル再訓練 API
@router.post("/retrain_model/")
async def retrain_model(
    input_data: PredictionInput = Body(...),
    db: Session = Depends(get_db)
):
    """モデルを再訓練します。"""
    # モデルを再訓練する処理を実装
    return {"message": "Model retraining started."}