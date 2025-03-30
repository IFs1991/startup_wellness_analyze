# 分析モジュールのテストパッケージ初期化ファイル

import sys
from unittest.mock import MagicMock

# 必要なモジュールとクラスをモック化
class MockAnalyzer(MagicMock):
    """分析クラスのベースモックです"""
    pass

# パッケージモジュールのモックを作成
for module_name in [
    'backend.analysis.BayesianInferenceAnalyzer',
    'backend.analysis.CausalInferenceAnalyzer',
    'backend.analysis.ClusterAnalyzer',
    'backend.analysis.FinancialAnalyzer',
    'backend.analysis.KnowledgeTransferIndexCalculator',
    'backend.analysis.MonteCarloSimulator',
    'backend.analysis.PortfolioNetworkAnalyzer',
    'backend.analysis.PredictiveModelAnalyzer',
    'backend.analysis.StartupSurvivabilityAnalyzer',
    'backend.analysis.Team_Analyzer'
]:
    # モジュールがまだモック化されていない場合のみモック化
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()
        # クラス名を抽出（最後のドット以降）
        class_name = module_name.split('.')[-1]
        # クラスのモックをモジュールに追加
        setattr(sys.modules[module_name], class_name, MockAnalyzer)