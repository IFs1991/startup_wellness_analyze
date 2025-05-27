"""
差分プライバシー統合コーディネーター

このモジュールは、RDPAccountant、AdaptiveClipping、PrivacyBudgetManagerの
3つのコンポーネントを統合し、統一されたインターフェースを提供します。

TDD Phase: GREEN - 統合テストを通すための実装
Task: dp_2.4 - 差分プライバシー統合テスト
"""

import asyncio
import uuid
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import time

from .rdp_accountant import RDPAccountant
from .adaptive_clipping import AdaptiveClipping
from .privacy_budget_manager import PrivacyBudgetManager, PrivacyBudgetExhaustionError

logger = logging.getLogger(__name__)

@dataclass
class IntegratedTrainingSession:
    """統合学習セッション情報"""
    session_id: str
    model_id: str
    config: Dict[str, Any]
    rdp_accountant: RDPAccountant
    adaptive_clipping: AdaptiveClipping
    budget_session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    total_rounds_completed: int = 0
    total_privacy_cost: float = 0.0

@dataclass
class IntegrationMetrics:
    """統合システムのメトリクス"""
    total_sessions: int = 0
    total_privacy_cost: float = 0.0
    average_clipping_norm: float = 0.0
    total_execution_time: float = 0.0
    component_overheads: Dict[str, float] = field(default_factory=dict)

class DifferentialPrivacyCoordinator:
    """差分プライバシー統合コーディネーター

    RDPAccountant、AdaptiveClipping、PrivacyBudgetManagerを統合し、
    エンドツーエンドの差分プライバシー保護を提供します。
    """

    def __init__(self,
                 total_epsilon: float = 10.0,
                 total_delta: float = 1e-5,
                 time_horizon: Optional[timedelta] = None,
                 enable_audit: bool = True,
                 enable_adaptive_clipping: bool = True):
        """初期化

        Args:
            total_epsilon: 総ε予算
            total_delta: 総δ予算
            time_horizon: 予算の有効期間
            enable_audit: 監査ログ有効化
            enable_adaptive_clipping: 適応的クリッピング有効化
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.time_horizon = time_horizon or timedelta(days=30)
        self.enable_audit = enable_audit
        self.enable_adaptive_clipping = enable_adaptive_clipping

        # コンポーネント初期化
        self.budget_manager = PrivacyBudgetManager(
            total_epsilon=total_epsilon,
            total_delta=total_delta,
            time_horizon=time_horizon,
            enable_audit=enable_audit
        )

        # アクティブセッション管理
        self.active_sessions: Dict[str, IntegratedTrainingSession] = {}
        self._lock = threading.RLock()

        # メトリクス
        self.metrics = IntegrationMetrics()
        self.created_at = datetime.now()

        logger.info(f"DifferentialPrivacyCoordinator初期化完了: ε={total_epsilon}, δ={total_delta}")

    def start_federated_training(self, config: Dict[str, Any]) -> str:
        """連合学習セッション開始

        Args:
            config: 学習設定

        Returns:
            統合セッションID

        Raises:
            PrivacyBudgetExhaustionError: 予算不足の場合
        """
        with self._lock:
            session_id = str(uuid.uuid4())
            model_id = config['model_id']

            # プライバシー予算マネージャーでセッション開始
            budget_session_id = self.budget_manager.start_training_session(config)

            # RDPアカウンタント初期化
            rdp_accountant = RDPAccountant()

            # 適応的クリッピング初期化（有効な場合）
            adaptive_clipping = None
            if self.enable_adaptive_clipping:
                adaptive_clipping = AdaptiveClipping(
                    initial_clipping_norm=config.get('clipping_norm', 1.0),
                    target_quantile=config.get('target_quantile', 0.5),
                    learning_rate=config.get('adaptive_learning_rate', 0.2)
                )

            # 統合セッション作成
            integrated_session = IntegratedTrainingSession(
                session_id=session_id,
                model_id=model_id,
                config=config,
                rdp_accountant=rdp_accountant,
                adaptive_clipping=adaptive_clipping,
                budget_session_id=budget_session_id
            )

            self.active_sessions[session_id] = integrated_session
            self.metrics.total_sessions += 1

            if self.enable_audit:
                self._log_integration_event('federated_training_start', {
                    'session_id': session_id,
                    'model_id': model_id,
                    'budget_session_id': budget_session_id,
                    'adaptive_clipping_enabled': self.enable_adaptive_clipping
                })

            logger.info(f"連合学習セッション開始: {session_id}, model={model_id}")
            return session_id

    def process_training_round(self,
                             session_id: str,
                             round_number: int,
                             gradients: np.ndarray,
                             participating_clients: int) -> Dict[str, Any]:
        """学習ラウンド処理

        Args:
            session_id: セッションID
            round_number: ラウンド番号
            gradients: 勾配データ
            participating_clients: 参加クライアント数

        Returns:
            処理結果（クリッピングされた勾配、プライバシーコスト等）

        Raises:
            ValueError: セッションが存在しない場合
            PrivacyBudgetExhaustionError: 予算超過の場合
        """
        with self._lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"セッションが存在しません: {session_id}")

            session = self.active_sessions[session_id]
            start_time = time.time()

            # 1. 適応的クリッピング適用
            clipped_gradients = gradients
            actual_clipping_norm = session.config.get('clipping_norm', 1.0)

            if session.adaptive_clipping:
                clipped_gradients, actual_clipping_norm = session.adaptive_clipping.clip_gradients(
                    gradients, session.config.get('clipping_norm')
                )

                # 適応的更新
                gradient_norm = np.linalg.norm(gradients)
                session.adaptive_clipping.update_clipping_norm(gradient_norm)

            # 2. プライバシー予算記録
            step_cost = self.budget_manager.record_training_step(
                session_id=session.budget_session_id,
                round_number=round_number,
                actual_noise_multiplier=session.config['noise_multiplier'],
                actual_clipping_norm=actual_clipping_norm,
                actual_sample_rate=session.config['sample_rate'],
                participating_clients=participating_clients
            )

            # 3. セッション状態更新
            session.total_rounds_completed += 1
            session.total_privacy_cost += step_cost['epsilon']

            # 4. メトリクス更新
            self.metrics.total_privacy_cost += step_cost['epsilon']
            processing_time = time.time() - start_time
            self.metrics.total_execution_time += processing_time

            # クリッピング閾値の平均更新
            if session.adaptive_clipping:
                current_avg = self.metrics.average_clipping_norm
                total_rounds = sum(s.total_rounds_completed for s in self.active_sessions.values())
                self.metrics.average_clipping_norm = (
                    (current_avg * (total_rounds - 1) + actual_clipping_norm) / total_rounds
                )

            result = {
                'clipped_gradients': clipped_gradients,
                'actual_clipping_norm': actual_clipping_norm,
                'privacy_cost': step_cost,
                'processing_time': processing_time,
                'round_number': round_number,
                'session_id': session_id
            }

            if self.enable_audit:
                self._log_integration_event('training_round_processed', {
                    'session_id': session_id,
                    'round_number': round_number,
                    'privacy_cost': step_cost,
                    'clipping_norm': actual_clipping_norm,
                    'processing_time': processing_time
                })

            logger.debug(f"学習ラウンド処理完了: session={session_id}, round={round_number}, ε={step_cost['epsilon']:.6f}")

            return result

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """セッション状態取得

        Args:
            session_id: セッションID

        Returns:
            セッション状態情報

        Raises:
            ValueError: セッションが存在しない場合
        """
        with self._lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"セッションが存在しません: {session_id}")

            session = self.active_sessions[session_id]

            # プライバシー予算マネージャーからの詳細情報
            budget_report = self.budget_manager.get_session_report(session.budget_session_id)

            # 適応的クリッピングの統計
            adaptive_stats = {}
            if session.adaptive_clipping:
                adaptive_stats = session.adaptive_clipping.get_statistics()

            return {
                'session_id': session_id,
                'model_id': session.model_id,
                'total_rounds_completed': session.total_rounds_completed,
                'total_privacy_cost': session.total_privacy_cost,
                'budget_report': budget_report,
                'adaptive_clipping_stats': adaptive_stats,
                'created_at': session.created_at.isoformat(),
                'config': session.config
            }

    def get_remaining_budget(self) -> Dict[str, float]:
        """残り予算取得

        Returns:
            残り予算情報
        """
        return self.budget_manager.get_remaining_budget()

    def get_integration_metrics(self) -> Dict[str, Any]:
        """統合システムメトリクス取得

        Returns:
            統合システムの包括的メトリクス
        """
        with self._lock:
            remaining_budget = self.get_remaining_budget()

            # コンポーネント別オーバーヘッド分析
            component_overheads = self._analyze_component_overheads()

            return {
                'total_sessions': self.metrics.total_sessions,
                'active_sessions': len(self.active_sessions),
                'total_privacy_cost': self.metrics.total_privacy_cost,
                'average_clipping_norm': self.metrics.average_clipping_norm,
                'total_execution_time': self.metrics.total_execution_time,
                'remaining_budget': remaining_budget,
                'component_overheads': component_overheads,
                'uptime': (datetime.now() - self.created_at).total_seconds(),
                'budget_usage_ratio': self.budget_manager.get_budget_usage_ratio()
            }

    def validate_privacy_guarantees(self, session_id: str) -> Dict[str, Any]:
        """プライバシー保証検証

        Args:
            session_id: セッションID

        Returns:
            プライバシー保証検証結果

        Raises:
            ValueError: セッションが存在しない場合
        """
        with self._lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"セッションが存在しません: {session_id}")

            session = self.active_sessions[session_id]

            # RDPアカウンタントでの独立検証
            rdp_values = session.rdp_accountant.compute_rdp(
                q=session.config['sample_rate'],
                noise_multiplier=session.config['noise_multiplier'],
                steps=session.total_rounds_completed,
                orders=session.rdp_accountant.orders
            )

            independent_epsilon = session.rdp_accountant.get_privacy_spent(
                orders=session.rdp_accountant.orders,
                rdp=rdp_values,
                target_delta=session.config.get('target_delta', self.total_delta / 10)
            )

            # プライバシー予算マネージャーからの証明
            privacy_proof = self.budget_manager.generate_privacy_proof(session.budget_session_id)

            # 一貫性検証
            epsilon_difference = abs(independent_epsilon - session.total_privacy_cost)
            relative_error = epsilon_difference / max(independent_epsilon, session.total_privacy_cost)

            is_consistent = relative_error < 0.1  # 10%以内の誤差
            is_valid = privacy_proof['is_valid'] and is_consistent

            return {
                'session_id': session_id,
                'is_valid': is_valid,
                'is_consistent': is_consistent,
                'independent_epsilon': independent_epsilon,
                'tracked_epsilon': session.total_privacy_cost,
                'relative_error': relative_error,
                'privacy_proof': privacy_proof,
                'validation_timestamp': datetime.now().isoformat()
            }

    def benchmark_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """パフォーマンスベンチマーク実行

        Args:
            config: ベンチマーク設定

        Returns:
            ベンチマーク結果
        """
        benchmark_config = {
            'model_id': f'benchmark_{uuid.uuid4().hex[:8]}',
            'client_count': config.get('clients', 50),
            'rounds': config.get('rounds', 10),
            'noise_multiplier': config.get('noise_multiplier', 1.1),
            'clipping_norm': config.get('clipping_norm', 1.0),
            'sample_rate': config.get('sample_rate', 0.01)
        }

        start_time = time.time()

        # ベンチマーク実行
        session_id = self.start_federated_training(benchmark_config)

        round_times = []
        for round_num in range(benchmark_config['rounds']):
            round_start = time.time()

            # ダミー勾配生成
            gradients = np.random.randn(config.get('model_dim', 784))

            # ラウンド処理
            result = self.process_training_round(
                session_id=session_id,
                round_number=round_num,
                gradients=gradients,
                participating_clients=benchmark_config['client_count']
            )

            round_times.append(time.time() - round_start)

        total_time = time.time() - start_time

        # セッション終了
        session_status = self.get_session_status(session_id)

        return {
            'config': benchmark_config,
            'total_execution_time': total_time,
            'average_round_time': np.mean(round_times),
            'rounds_per_second': benchmark_config['rounds'] / total_time,
            'time_per_client': total_time / benchmark_config['client_count'],
            'total_privacy_cost': session_status['total_privacy_cost'],
            'round_times': round_times,
            'session_status': session_status
        }

    def _analyze_component_overheads(self) -> Dict[str, float]:
        """コンポーネント別オーバーヘッド分析"""
        # 簡易的なオーバーヘッド分析
        # 実際の実装では、より詳細なプロファイリングを行う
        total_time = self.metrics.total_execution_time
        total_sessions = max(self.metrics.total_sessions, 1)

        estimated_overheads = {
            'rdp_accountant': total_time * 0.3 / total_sessions,  # 30%
            'adaptive_clipping': total_time * 0.2 / total_sessions,  # 20%
            'privacy_budget_manager': total_time * 0.4 / total_sessions,  # 40%
            'coordination_overhead': total_time * 0.1 / total_sessions  # 10%
        }

        return estimated_overheads

    def _log_integration_event(self, action: str, details: Dict[str, Any]):
        """統合イベントのログ記録"""
        if self.enable_audit:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'component': 'differential_privacy_coordinator',
                **details
            }
            # 実際の実装では、専用のログストレージに記録
            logger.info(f"Integration event: {action} - {details}")

    def close_session(self, session_id: str) -> Dict[str, Any]:
        """セッション終了

        Args:
            session_id: セッションID

        Returns:
            セッション終了レポート

        Raises:
            ValueError: セッションが存在しない場合
        """
        with self._lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"セッションが存在しません: {session_id}")

            session = self.active_sessions[session_id]

            # 最終レポート生成
            final_report = self.get_session_status(session_id)
            privacy_validation = self.validate_privacy_guarantees(session_id)

            # セッション削除
            del self.active_sessions[session_id]

            if self.enable_audit:
                self._log_integration_event('session_closed', {
                    'session_id': session_id,
                    'final_report': final_report,
                    'privacy_validation': privacy_validation
                })

            logger.info(f"セッション終了: {session_id}")

            return {
                'session_id': session_id,
                'final_report': final_report,
                'privacy_validation': privacy_validation,
                'closed_at': datetime.now().isoformat()
            }

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """全セッション情報取得

        Returns:
            全アクティブセッションの情報
        """
        with self._lock:
            return [
                self.get_session_status(session_id)
                for session_id in self.active_sessions.keys()
            ]