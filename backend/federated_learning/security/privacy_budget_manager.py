"""
プライバシー予算管理API実装

このモジュールは、差分プライバシーにおけるプライバシー予算の管理を実装します。
複数モデルの学習計画に対する予算配分、消費追跡、枯渇処理を統合的に提供します。

TDD Phase: GREEN - テストを通す実装
"""

import asyncio
import uuid
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import math

from .rdp_accountant import RDPAccountant
from .adaptive_clipping import AdaptiveClipping

logger = logging.getLogger(__name__)

@dataclass
class BudgetAllocation:
    """予算配分情報"""
    epsilon: float
    delta: float
    model_id: str
    priority: float = 1.0
    allocated_at: datetime = field(default_factory=datetime.now)

@dataclass
class TrainingSession:
    """学習セッション情報"""
    session_id: str
    model_id: str
    config: Dict[str, Any]
    allocated_budget: BudgetAllocation
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    steps_recorded: int = 0

class PrivacyBudgetExhaustionError(Exception):
    """プライバシー予算枯渇エラー"""
    pass

class PrivacyBudgetManager:
    """プライバシー予算マネージャー

    差分プライバシーにおけるプライバシー予算の包括的管理を実装します。
    複数モデルの学習に対する予算配分、消費追跡、枯渇処理を提供します。
    """

    def __init__(self,
                 total_epsilon: float = 10.0,
                 total_delta: float = 1e-5,
                 time_horizon: Optional[timedelta] = None,
                 alert_threshold: float = 0.8,
                 enable_audit: bool = True):
        """初期化

        Args:
            total_epsilon: 総ε予算
            total_delta: 総δ予算
            time_horizon: 予算の有効期間
            alert_threshold: アラート発生閾値（予算使用率）
            enable_audit: 監査ログ有効化

        Raises:
            ValueError: パラメータが無効な場合
        """
        # パラメータ検証
        if total_epsilon <= 0:
            raise ValueError(f"ε予算は正の値である必要があります: {total_epsilon}")
        if not (0 < total_delta < 1):
            raise ValueError(f"δは0 < δ < 1である必要があります: {total_delta}")
        if time_horizon and time_horizon.total_seconds() <= 0:
            raise ValueError(f"時間軸は正の値である必要があります: {time_horizon}")
        if not (0 < alert_threshold <= 1):
            raise ValueError(f"アラート閾値は0 < threshold ≤ 1である必要があります: {alert_threshold}")

        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.time_horizon = time_horizon or timedelta(days=30)
        self.alert_threshold = alert_threshold
        self.enable_audit = enable_audit

        # 予算管理
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
        self.allocated_budgets: Dict[str, BudgetAllocation] = {}
        self.active_sessions: Dict[str, TrainingSession] = {}

        # 同期制御
        self._lock = threading.RLock()

        # コンポーネント
        self.rdp_accountant = RDPAccountant()

        # 監査とアラート
        self.audit_logs: List[Dict[str, Any]] = []
        self.alert_callback: Optional[Callable] = None

        # 統計
        self.created_at = datetime.now()
        self.last_allocation_at: Optional[datetime] = None

        logger.info(f"PrivacyBudgetManager初期化完了: ε={total_epsilon}, δ={total_delta}, horizon={time_horizon}")

    def allocate_budget(self, learning_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """プライバシー予算配分

        Args:
            learning_configs: 学習設定のリスト

        Returns:
            モデルIDをキーとする予算配分辞書

        Raises:
            PrivacyBudgetExhaustionError: 予算不足の場合
        """
        with self._lock:
            # 必要予算を計算
            required_budgets = self._estimate_required_budgets(learning_configs)

            # 利用可能予算を確認
            available_epsilon = self.total_epsilon - self.consumed_epsilon
            available_delta = self.total_delta - self.consumed_delta

            total_required_epsilon = sum(budget['epsilon'] for budget in required_budgets.values())
            total_required_delta = sum(budget['delta'] for budget in required_budgets.values())

            if total_required_epsilon > available_epsilon or total_required_delta > available_delta:
                raise PrivacyBudgetExhaustionError(
                    f"予算不足: required_ε={total_required_epsilon}, available_ε={available_epsilon}, "
                    f"required_δ={total_required_delta}, available_δ={available_delta}"
                )

            # 予算配分の最適化
            allocations = self._optimize_budget_allocation(learning_configs, required_budgets)

            # 配分情報を記録
            self.last_allocation_at = datetime.now()

            if self.enable_audit:
                self._log_audit_event('budget_allocation', {
                    'total_models': len(learning_configs),
                    'allocated_epsilon': sum(alloc['epsilon'] for alloc in allocations.values()),
                    'allocated_delta': sum(alloc['delta'] for alloc in allocations.values()),
                    'allocations': allocations
                })

            logger.info(f"予算配分完了: {len(learning_configs)}モデル, total_ε={total_required_epsilon:.4f}")
            return allocations

    def _estimate_required_budgets(self, learning_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """必要予算の推定"""
        required_budgets = {}

        for config in learning_configs:
            model_id = config['model_id']

            # RDPアカウンタントで必要予算を計算
            rdp_values = self.rdp_accountant.compute_rdp(
                q=config['sample_rate'],
                noise_multiplier=config['noise_multiplier'],
                steps=config['rounds'],
                orders=self.rdp_accountant.orders
            )

            epsilon = self.rdp_accountant.get_privacy_spent(
                orders=self.rdp_accountant.orders,
                rdp=rdp_values,
                target_delta=self.total_delta / len(learning_configs)  # 均等分割
            )

            required_budgets[model_id] = {
                'epsilon': epsilon,
                'delta': self.total_delta / len(learning_configs)
            }

        return required_budgets

    def _optimize_budget_allocation(self,
                                  learning_configs: List[Dict[str, Any]],
                                  required_budgets: Dict[str, Dict[str, float]],
                                  priorities: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """予算配分の最適化"""
        if priorities is None:
            # デフォルト優先度（均等）
            priorities = {config['model_id']: 1.0 for config in learning_configs}

        # 優先度に基づく重み付け配分
        total_priority = sum(priorities.values())
        allocations = {}

        available_epsilon = self.total_epsilon - self.consumed_epsilon
        available_delta = self.total_delta - self.consumed_delta

        # 必要予算の合計を計算
        total_required_epsilon = sum(budget['epsilon'] for budget in required_budgets.values())
        total_required_delta = sum(budget['delta'] for budget in required_budgets.values())

        # スケーリング係数を計算（利用可能予算内に収める）
        epsilon_scale = min(1.0, available_epsilon * 0.5 / max(total_required_epsilon, 1e-10))  # 50%に変更（より保守的）
        delta_scale = min(1.0, available_delta * 0.5 / max(total_required_delta, 1e-10))  # 50%に変更（より保守的）

        for config in learning_configs:
            model_id = config['model_id']
            priority = priorities.get(model_id, 1.0)

            # 必要予算をベースとした配分
            required_epsilon = required_budgets[model_id]['epsilon']
            required_delta = required_budgets[model_id]['delta']

            # スケーリングを適用して利用可能予算内に収める
            allocated_epsilon = required_epsilon * epsilon_scale
            allocated_delta = required_delta * delta_scale

            # 最小予算の保証
            min_epsilon = 0.001  # 最小ε予算
            min_delta = 1e-8     # 最小δ予算

            allocations[model_id] = {
                'epsilon': max(min_epsilon, allocated_epsilon),
                'delta': max(min_delta, allocated_delta)
            }

        return allocations

    def compute_allocation_efficiency(self,
                                    allocations: Dict[str, Dict[str, float]],
                                    learning_configs: List[Dict[str, Any]]) -> float:
        """配分効率性の計算"""
        # パレート効率性の簡易指標
        total_allocated_epsilon = sum(alloc['epsilon'] for alloc in allocations.values())
        available_epsilon = self.total_epsilon - self.consumed_epsilon

        utilization_rate = total_allocated_epsilon / available_epsilon

        # 配分の均衡性を評価
        epsilon_values = [alloc['epsilon'] for alloc in allocations.values()]
        coefficient_of_variation = np.std(epsilon_values) / np.mean(epsilon_values) if np.mean(epsilon_values) > 0 else 0

        # 効率性スコア（0-1）
        efficiency_score = utilization_rate * (1 - coefficient_of_variation * 0.5)

        return min(1.0, max(0.0, efficiency_score))

    def request_budget(self, learning_request: Dict[str, Any]) -> str:
        """予算リクエスト処理

        Args:
            learning_request: 学習リクエスト

        Returns:
            セッションID

        Raises:
            PrivacyBudgetExhaustionError: 予算不足の場合
        """
        # 単一リクエストを配列として処理
        allocations = self.allocate_budget([learning_request])
        model_id = learning_request['model_id']

        # セッション開始
        return self.start_training_session(learning_request, allocations[model_id])

    def suggest_feasible_plan(self, learning_request: Dict[str, Any]) -> Dict[str, Any]:
        """実行可能な学習計画の提案

        Args:
            learning_request: 元の学習リクエスト

        Returns:
            実行可能な学習計画
        """
        available_epsilon = self.total_epsilon - self.consumed_epsilon
        available_delta = self.total_delta - self.consumed_delta

        # パラメータを調整して実行可能な計画を作成
        suggested_plan = learning_request.copy()

        # ノイズ乗数を増加させてプライバシーコストを削減
        noise_multiplier = learning_request.get('noise_multiplier', 1.1)
        while noise_multiplier < 5.0:  # 上限
            # 調整された設定でプライバシーコストを計算
            rdp_values = self.rdp_accountant.compute_rdp(
                q=learning_request['sample_rate'],
                noise_multiplier=noise_multiplier,
                steps=learning_request['rounds'],
                orders=self.rdp_accountant.orders
            )

            epsilon = self.rdp_accountant.get_privacy_spent(
                orders=self.rdp_accountant.orders,
                rdp=rdp_values,
                target_delta=available_delta * 0.5
            )

            if epsilon <= available_epsilon * 0.9:  # 90%以内に収める
                suggested_plan.update({
                    'noise_multiplier': noise_multiplier,
                    'estimated_epsilon': epsilon,
                    'estimated_delta': available_delta * 0.5,
                    'feasible': True
                })
                break

            noise_multiplier += 0.1
        else:
            # ラウンド数を削減
            max_rounds = int(learning_request['rounds'] * available_epsilon / (available_epsilon + 1))
            suggested_plan.update({
                'rounds': max(1, max_rounds),
                'estimated_epsilon': available_epsilon * 0.9,
                'estimated_delta': available_delta * 0.5,
                'feasible': True
            })

        return suggested_plan

    def start_training_session(self,
                             config: Dict[str, Any],
                             allocated_budget: Optional[Dict[str, float]] = None) -> str:
        """学習セッション開始

        Args:
            config: 学習設定
            allocated_budget: 配分された予算

        Returns:
            セッションID
        """
        with self._lock:
            session_id = str(uuid.uuid4())
            model_id = config['model_id']

            if allocated_budget is None:
                # より安全な固定予算配分を実行
                available_epsilon = self.total_epsilon - self.consumed_epsilon
                available_delta = self.total_delta - self.consumed_delta

                # 非常に保守的な固定予算（利用可能予算の10%）
                allocated_budget = {
                    'epsilon': min(1.0, available_epsilon * 0.1),  # 利用可能予算の10%または最大1.0
                    'delta': min(1e-4, available_delta * 0.1)      # 利用可能予算の10%または最大1e-4
                }

            # セッション作成
            budget_allocation = BudgetAllocation(
                epsilon=allocated_budget['epsilon'],
                delta=allocated_budget['delta'],
                model_id=model_id
            )

            session = TrainingSession(
                session_id=session_id,
                model_id=model_id,
                config=config,
                allocated_budget=budget_allocation
            )

            self.active_sessions[session_id] = session

            if self.enable_audit:
                self._log_audit_event('session_start', {
                    'session_id': session_id,
                    'model_id': model_id,
                    'allocated_epsilon': allocated_budget['epsilon'],
                    'allocated_delta': allocated_budget['delta']
                })

            logger.info(f"学習セッション開始: {session_id}, model={model_id}")
            return session_id

    def record_training_step(self,
                           session_id: str,
                           round_number: int,
                           actual_noise_multiplier: float,
                           actual_clipping_norm: float,
                           actual_sample_rate: float,
                           participating_clients: int) -> Dict[str, float]:
        """学習ステップの記録

        Args:
            session_id: セッションID
            round_number: ラウンド番号
            actual_noise_multiplier: 実際のノイズ乗数
            actual_clipping_norm: 実際のクリッピング閾値
            actual_sample_rate: 実際のサンプリング率
            participating_clients: 参加クライアント数

        Returns:
            このステップでの予算消費量

        Raises:
            ValueError: セッションが存在しない場合
            PrivacyBudgetExhaustionError: 予算超過の場合
        """
        with self._lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"セッションが存在しません: {session_id}")

            session = self.active_sessions[session_id]

            # 1ステップのプライバシーコストを計算
            rdp_values = self.rdp_accountant.compute_rdp(
                q=actual_sample_rate,
                noise_multiplier=actual_noise_multiplier,
                steps=1,
                orders=self.rdp_accountant.orders
            )

            step_epsilon = self.rdp_accountant.get_privacy_spent(
                orders=self.rdp_accountant.orders,
                rdp=rdp_values,
                target_delta=session.allocated_budget.delta / session.config['rounds']
            )

            step_delta = session.allocated_budget.delta / session.config['rounds']

            # 予算超過チェック
            if (session.consumed_epsilon + step_epsilon > session.allocated_budget.epsilon or
                session.consumed_delta + step_delta > session.allocated_budget.delta):
                raise PrivacyBudgetExhaustionError(
                    f"セッション予算超過: session={session_id}, "
                    f"consumed_ε={session.consumed_epsilon + step_epsilon}, allocated_ε={session.allocated_budget.epsilon}"
                )

            # 予算消費を記録
            session.consumed_epsilon += step_epsilon
            session.consumed_delta += step_delta
            session.steps_recorded += 1

            self.consumed_epsilon += step_epsilon
            self.consumed_delta += step_delta

            # アラートチェック
            self._check_budget_alerts()

            if self.enable_audit:
                self._log_audit_event('training_step', {
                    'session_id': session_id,
                    'round_number': round_number,
                    'epsilon_consumed': step_epsilon,
                    'delta_consumed': step_delta,
                    'participating_clients': participating_clients
                })

            logger.debug(f"学習ステップ記録: session={session_id}, round={round_number}, ε={step_epsilon:.6f}")

            return {
                'epsilon': step_epsilon,
                'delta': step_delta
            }

    def get_remaining_budget(self) -> Dict[str, float]:
        """残り予算の取得

        Returns:
            残り予算情報
        """
        with self._lock:
            return {
                'epsilon': self.total_epsilon - self.consumed_epsilon,
                'delta': self.total_delta - self.consumed_delta,
                'epsilon_ratio': (self.total_epsilon - self.consumed_epsilon) / self.total_epsilon,
                'delta_ratio': (self.total_delta - self.consumed_delta) / self.total_delta
            }

    def get_session_report(self, session_id: str) -> Dict[str, Any]:
        """セッションレポートの取得

        Args:
            session_id: セッションID

        Returns:
            セッション詳細情報

        Raises:
            ValueError: セッションが存在しない場合
        """
        with self._lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"セッションが存在しません: {session_id}")

            session = self.active_sessions[session_id]

            return {
                'session_id': session_id,
                'model_id': session.model_id,
                'consumed_epsilon': session.consumed_epsilon,
                'consumed_delta': session.consumed_delta,
                'allocated_epsilon': session.allocated_budget.epsilon,
                'allocated_delta': session.allocated_budget.delta,
                'steps_recorded': session.steps_recorded,
                'efficiency': session.consumed_epsilon / session.allocated_budget.epsilon if session.allocated_budget.epsilon > 0 else 0,
                'created_at': session.created_at.isoformat()
            }

    def optimize_allocation(self,
                          learning_configs: List[Dict[str, Any]],
                          priorities: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """優先度を考慮した予算配分最適化

        Args:
            learning_configs: 学習設定のリスト
            priorities: モデル優先度

        Returns:
            最適化された予算配分
        """
        required_budgets = self._estimate_required_budgets(learning_configs)
        return self._optimize_budget_allocation(learning_configs, required_budgets, priorities)

    def get_budget_usage_ratio(self) -> float:
        """予算使用率の取得

        Returns:
            予算使用率（0-1）
        """
        with self._lock:
            epsilon_usage = self.consumed_epsilon / self.total_epsilon
            delta_usage = self.consumed_delta / self.total_delta
            return max(epsilon_usage, delta_usage)

    def set_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """アラートコールバックの設定

        Args:
            callback: アラート時に呼び出されるコールバック関数
        """
        self.alert_callback = callback

    def _check_budget_alerts(self):
        """予算アラートのチェック"""
        usage_ratio = self.get_budget_usage_ratio()

        if usage_ratio >= self.alert_threshold and self.alert_callback:
            alert_data = {
                'budget_usage': usage_ratio,
                'consumed_epsilon': self.consumed_epsilon,
                'consumed_delta': self.consumed_delta,
                'total_epsilon': self.total_epsilon,
                'total_delta': self.total_delta,
                'timestamp': datetime.now().isoformat()
            }

            try:
                self.alert_callback(alert_data)
            except Exception as e:
                logger.error(f"アラートコールバック実行エラー: {e}")

    def get_audit_logs(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """監査ログの取得

        Args:
            session_id: 特定のセッションのログのみ取得する場合

        Returns:
            監査ログのリスト
        """
        if session_id:
            return [log for log in self.audit_logs if log.get('session_id') == session_id]
        return self.audit_logs.copy()

    def generate_privacy_proof(self, session_id: str) -> Dict[str, Any]:
        """プライバシー証明の生成

        Args:
            session_id: セッションID

        Returns:
            プライバシー証明書

        Raises:
            ValueError: セッションが存在しない場合
        """
        with self._lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"セッションが存在しません: {session_id}")

            session = self.active_sessions[session_id]
            session_logs = self.get_audit_logs(session_id)

            # プライバシー保証の検証
            total_epsilon = sum(log['epsilon_consumed'] for log in session_logs if 'epsilon_consumed' in log)
            total_delta = sum(log['delta_consumed'] for log in session_logs if 'delta_consumed' in log)

            is_valid = (
                total_epsilon <= self.total_epsilon and
                total_delta <= self.total_delta and
                total_epsilon == session.consumed_epsilon and
                total_delta == session.consumed_delta
            )

            return {
                'session_id': session_id,
                'model_id': session.model_id,
                'total_epsilon': total_epsilon,
                'total_delta': total_delta,
                'is_valid': is_valid,
                'verification_steps': len(session_logs),
                'generated_at': datetime.now().isoformat(),
                'proof_hash': hash(str(session_logs))  # 簡易整合性チェック
            }

    def _log_audit_event(self, action: str, details: Dict[str, Any]):
        """監査イベントのログ記録"""
        if self.enable_audit:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                **details
            }
            self.audit_logs.append(log_entry)

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報の取得

        Returns:
            プライバシー予算管理の統計情報
        """
        with self._lock:
            return {
                'total_epsilon': self.total_epsilon,
                'total_delta': self.total_delta,
                'consumed_epsilon': self.consumed_epsilon,
                'consumed_delta': self.consumed_delta,
                'remaining_epsilon': self.total_epsilon - self.consumed_epsilon,
                'remaining_delta': self.total_delta - self.consumed_delta,
                'usage_ratio': self.get_budget_usage_ratio(),
                'active_sessions': len(self.active_sessions),
                'total_audit_logs': len(self.audit_logs),
                'uptime': (datetime.now() - self.created_at).total_seconds(),
                'last_allocation_at': self.last_allocation_at.isoformat() if self.last_allocation_at else None
            }