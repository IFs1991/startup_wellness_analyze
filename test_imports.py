"""
モジュールインポートテスト
"""
import sys
import importlib

def test_imports():
    """モジュールのインポートをテストします"""
    # Pythonバージョンの表示
    print(f"Python バージョン: {sys.version}")

    # Flowerフレームワークのテスト
    try:
        import flwr
        print(f"Flowerバージョン: {flwr.__version__}")
        print("Flowerインポート: 成功")
    except Exception as e:
        print(f"Flowerインポートエラー: {e}")

    # コアモジュールのテスト
    try:
        from backend.core.wellness_score_calculator import WellnessScoreCalculator
        print("WellnessScoreCalculator インポート: 成功")
    except Exception as e:
        print(f"WellnessScoreCalculatorインポートエラー: {e}")

    try:
        from backend.core.performance_predictor import PerformancePredictor
        print("PerformancePredictor インポート: 成功")
    except Exception as e:
        print(f"PerformancePredictorインポートエラー: {e}")

    # 連合学習モジュールが利用可能かチェック（エラーは捕捉）
    try:
        import backend.federated_learning
        print("連合学習モジュールインポート: 成功")
        print(f"連合学習モジュールバージョン: {backend.federated_learning.__version__}")
    except Exception as e:
        print(f"連合学習モジュールインポートエラー: {e}")

if __name__ == "__main__":
    test_imports()