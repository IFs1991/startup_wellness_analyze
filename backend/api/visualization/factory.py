"""
可視化処理のファクトリーパターン実装

このモジュールでは、分析タイプに基づいて適切な可視化処理クラスを
生成するためのファクトリーパターンを実装します。
パフォーマンス最適化のための改善を含みます。
"""

from typing import Dict, Any, Type, Callable, Optional, Protocol, Union
import logging
import importlib
import weakref
from .errors import InvalidVisualizationTypeError

logger = logging.getLogger(__name__)


class VisualizationProcessor(Protocol):
    """可視化プロセッサのプロトコル定義"""

    def prepare_chart_data(self, analysis_results: Dict[str, Any],
                          visualization_type: str,
                          options: Dict[str, Any]) -> Dict[str, Any]:
        """
        チャートデータを準備する

        Args:
            analysis_results: 分析結果
            visualization_type: 可視化タイプ
            options: オプション設定

        Returns:
            チャートデータ
        """
        ...

    def format_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析結果のサマリーをフォーマットする

        Args:
            analysis_results: 分析結果

        Returns:
            フォーマット済みサマリー
        """
        ...


class VisualizationProcessorFactory:
    """可視化プロセッサのファクトリークラス（最適化版）"""

    # プロセッサクラスの登録情報
    # キーは分析タイプ、値はプロセッサクラスか、またはそのクラスをロードするための情報
    _processor_registry: Dict[str, Union[Type[VisualizationProcessor], Dict[str, str]]] = {}

    # プロセッサインスタンスのキャッシュ（弱参照を使用してメモリリークを防止）
    _processor_instances = weakref.WeakValueDictionary()

    @classmethod
    def register(cls, analysis_type: str, processor_class: Union[Type[VisualizationProcessor], Dict[str, str]]) -> None:
        """
        新しい可視化プロセッサクラスを登録する

        Args:
            analysis_type: 分析タイプ
            processor_class: プロセッサクラスまたはそのロード情報
        """
        cls._processor_registry[analysis_type] = processor_class
        logger.info(f"可視化プロセッサ登録: {analysis_type}")

    @classmethod
    def register_lazy(cls, analysis_type: str, module_path: str, class_name: str) -> None:
        """
        遅延ロード用の可視化プロセッサクラスを登録する

        Args:
            analysis_type: 分析タイプ
            module_path: モジュールパス
            class_name: クラス名
        """
        cls._processor_registry[analysis_type] = {
            "module_path": module_path,
            "class_name": class_name
        }
        logger.info(f"可視化プロセッサ遅延登録: {analysis_type} -> {module_path}.{class_name}")

    @classmethod
    def get_processor(cls, analysis_type: str) -> VisualizationProcessor:
        """
        指定された分析タイプのプロセッサインスタンスを取得する
        （最適化版：キャッシュとメモリ効率化を実装）

        Args:
            analysis_type: 分析タイプ

        Returns:
            プロセッサインスタンス

        Raises:
            InvalidVisualizationTypeError: 未登録の分析タイプが指定された場合
        """
        # 未登録の分析タイプのチェック
        if analysis_type not in cls._processor_registry:
            fallback_type = "generic"
            if fallback_type not in cls._processor_registry:
                logger.error(f"未登録の分析タイプ: {analysis_type}")
                raise InvalidVisualizationTypeError(
                    f"分析タイプ '{analysis_type}' は登録されていません。"
                    f"利用可能なタイプ: {', '.join(cls._processor_registry.keys())}"
                )
            logger.warning(f"未登録の分析タイプ: {analysis_type}、汎用プロセッサを使用します")
            analysis_type = fallback_type

        # キャッシュからインスタンスを取得
        if analysis_type in cls._processor_instances:
            return cls._processor_instances[analysis_type]

        # 登録情報を取得
        processor_info = cls._processor_registry[analysis_type]

        # プロセッサクラスを取得（遅延ロードの場合はロードを実行）
        processor_class = None
        if isinstance(processor_info, dict):
            # 遅延ロード情報から動的にクラスをロード
            module_path = processor_info["module_path"]
            class_name = processor_info["class_name"]
            try:
                module = importlib.import_module(module_path)
                processor_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"プロセッサクラスのロードに失敗: {module_path}.{class_name} - {str(e)}")
                # 汎用プロセッサにフォールバック
                if "generic" in cls._processor_registry:
                    processor_class = cls._processor_registry["generic"]
                    if isinstance(processor_class, dict):
                        # 汎用プロセッサも遅延ロードの場合
                        generic_module_path = processor_class["module_path"]
                        generic_class_name = processor_class["class_name"]
                        try:
                            module = importlib.import_module(generic_module_path)
                            processor_class = getattr(module, generic_class_name)
                        except (ImportError, AttributeError) as e:
                            logger.error(f"汎用プロセッサクラスのロードに失敗: {generic_module_path}.{generic_class_name} - {str(e)}")
                            raise InvalidVisualizationTypeError(f"プロセッサクラスのロードに失敗しました")
                else:
                    raise InvalidVisualizationTypeError(f"プロセッサクラスのロードに失敗しました")
        else:
            # 直接クラスが登録されている場合
            processor_class = processor_info

        # インスタンスを生成してキャッシュに保存
        processor = processor_class()
        cls._processor_instances[analysis_type] = processor

        logger.debug(f"プロセッサインスタンス生成: {analysis_type} -> {processor.__class__.__name__}")
        return processor

    @classmethod
    def list_registered_processors(cls) -> Dict[str, str]:
        """
        登録されているプロセッサの一覧を取得する

        Returns:
            {分析タイプ: プロセッサクラス名} の辞書
        """
        result = {}
        for analysis_type, processor_info in cls._processor_registry.items():
            if isinstance(processor_info, dict):
                result[analysis_type] = f"{processor_info['module_path']}.{processor_info['class_name']}"
            else:
                result[analysis_type] = processor_info.__name__
        return result


# デコレータを使った登録の簡略化
def register_processor(analysis_type: str) -> Callable[[Type[VisualizationProcessor]], Type[VisualizationProcessor]]:
    """
    可視化プロセッサをファクトリーに登録するデコレータ

    Args:
        analysis_type: 分析タイプ名

    Returns:
        登録用デコレータ
    """
    def decorator(processor_class: Type[VisualizationProcessor]) -> Type[VisualizationProcessor]:
        VisualizationProcessorFactory.register(analysis_type, processor_class)
        return processor_class

    return decorator


# 汎用プロセッサの実装
class GenericVisualizationProcessor:
    """汎用可視化プロセッサ（フォールバック用）"""

    def prepare_chart_data(self, analysis_results: Dict[str, Any],
                          visualization_type: str,
                          options: Dict[str, Any]) -> Dict[str, Any]:
        """
        チャートデータを準備する汎用実装

        Args:
            analysis_results: 分析結果
            visualization_type: 可視化タイプ
            options: オプション設定

        Returns:
            チャートデータ
        """
        # 最低限のデータ準備（メモリ効率を考慮）
        # 大きなデータセットを扱う場合の最適化
        if isinstance(analysis_results.get("data"), list) and len(analysis_results.get("data", [])) > 1000:
            # 大きなデータセットはサンプリング
            data = analysis_results.get("data", [])[::10]  # 10件ごとにサンプリング
        else:
            data = analysis_results.get("data", [])

        return {
            "config": {
                "chart": {
                    "type": visualization_type
                },
                "title": {
                    "text": options.get("title", "分析結果")
                }
            },
            "data": {
                "values": data
            }
        }

    def format_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析結果のサマリーをフォーマットする汎用実装

        Args:
            analysis_results: 分析結果

        Returns:
            フォーマット済みサマリー
        """
        # 基本的なサマリー情報を抽出（メモリ効率を考慮）
        if isinstance(analysis_results, dict):
            # 巨大なデータを除外
            return {
                "summary": "基本分析結果",
                "key_metrics": {
                    k: v for k, v in analysis_results.items()
                    if not isinstance(v, (list, dict)) and not k.startswith("_")
                }
            }
        return {"summary": "分析結果データが有効な形式ではありません"}


# 汎用プロセッサを登録
VisualizationProcessorFactory.register("generic", GenericVisualizationProcessor)