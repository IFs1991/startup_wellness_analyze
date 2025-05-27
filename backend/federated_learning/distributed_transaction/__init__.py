# Phase 3 Task 3.4: 分散トランザクション管理システムの実装
# パッケージ定義とエクスポート

"""
分散トランザクション管理システム

このパッケージは、フェデレーテッド学習環境での分散トランザクション管理を提供します。

主要コンポーネント:
- DistributedTransactionManager: 中央管理マネージャー
- TwoPhaseCommitCoordinator: 2相コミットプロトコル
- SagaCoordinator: Sagaパターン実装
- CompensationEngine: 補償エンジン

使用例:
```python
from federated_learning.distributed_transaction import (
    DistributedTransactionManager,
    Transaction,
    TransactionStep
)

# トランザクションマネージャー初期化
manager = DistributedTransactionManager()

# トランザクション作成
transaction = manager.create_transaction(
    transaction_type="saga",
    consistency_level="eventual"
)

# ステップ追加
step = TransactionStep(
    step_id="step_1",
    operation="update_model",
    resource_id="model_registry",
    data={"model_id": "model_123", "version": "1.1.0"},
    compensation_data={"model_id": "model_123", "version": "1.0.0"}
)
transaction.steps.append(step)

# トランザクション実行
result = await manager.execute_transaction(transaction)
```
"""

from .models import (
    # Enums
    TransactionStatus,
    StepStatus,
    IsolationLevel,
    ConsistencyLevel,

    # Data Models
    TransactionStep,
    CompensationAction,
    Transaction,
    SagaTransaction,
    TwoPhaseCommitResult,
    SagaResult,
    CompensationResult,
    ResourceManager,
    DistributedLock,
    TransactionLog,

    # Utility Functions
    generate_transaction_id,
    generate_step_id,
    generate_saga_id,
    generate_compensation_id,
    is_terminal_status,
    can_be_compensated,
    calculate_transaction_duration,
    get_step_by_id,
    get_failed_steps,
    get_completed_steps,
    validate_transaction
)

from .transaction_manager import DistributedTransactionManager
from .two_phase_commit import TwoPhaseCommitCoordinator
from .saga_coordinator import SagaCoordinator
from .compensation_engine import CompensationEngine

__all__ = [
    # Main Components
    "DistributedTransactionManager",
    "TwoPhaseCommitCoordinator",
    "SagaCoordinator",
    "CompensationEngine",

    # Enums
    "TransactionStatus",
    "StepStatus",
    "IsolationLevel",
    "ConsistencyLevel",

    # Data Models
    "TransactionStep",
    "CompensationAction",
    "Transaction",
    "SagaTransaction",
    "TwoPhaseCommitResult",
    "SagaResult",
    "CompensationResult",
    "ResourceManager",
    "DistributedLock",
    "TransactionLog",

    # Utility Functions
    "generate_transaction_id",
    "generate_step_id",
    "generate_saga_id",
    "generate_compensation_id",
    "is_terminal_status",
    "can_be_compensated",
    "calculate_transaction_duration",
    "get_step_by_id",
    "get_failed_steps",
    "get_completed_steps",
    "validate_transaction"
]

__version__ = "1.0.0"
__author__ = "Federated Learning Team"
__description__ = "Distributed Transaction Management System for Federated Learning"