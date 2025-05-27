"""
セキュア集約プロトコルのテスト

TDD Phase 1, Task 1.2: セキュア集約プロトコルの完全実装
RED段階: 失敗するテストを最初に書く

TDD.yamlに基づく要件:
- test_secure_aggregation_with_dropout: ドロップアウトありセキュア集約
- test_malicious_client_detection: 悪意のあるクライアント検出
- test_aggregation_correctness: 集約の正確性
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch, AsyncMock
import asyncio

class TestSecureAggregation:
    """セキュア集約プロトコルのテストクラス

    TDD.yamlに基づく要件実装:
    - マスキング機構のテスト
    - シークレット共有のテスト
    - 検証可能な集約のテスト
    - ドロップアウト処理のテスト
    - 悪意のあるクライアント検出のテスト
    """

    def test_secure_aggregation_with_dropout(self):
        """ドロップアウトありセキュア集約のテスト

        要件:
        - 一部のクライアントがドロップアウトしても集約が続行できること
        - ドロップアウト閾値を下回った場合は集約を中止すること
        - ドロップアウトしたクライアントのマスクを適切に除去できること
        """
        # まだSecureAggregationProtocolクラスが実装されていないため、このテストは失敗する（RED段階）
        from backend.federated_learning.security.secure_aggregation_protocol import SecureAggregationProtocol

        # テストパラメータ
        total_clients = 5
        min_clients = 3
        dropout_clients = [2, 4]  # クライアント2と4がドロップアウト

        protocol = SecureAggregationProtocol(
            total_clients=total_clients,
            min_clients=min_clients,
            key_size=1024  # テスト用に小さなキーサイズ
        )

        # クライアントのモデル更新を準備
        client_updates = {}
        for client_id in range(total_clients):
            if client_id not in dropout_clients:
                # 実際の更新値（テスト用に小さな値）
                update = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
                client_updates[client_id] = update

        # セキュア集約を実行
        aggregated_result = protocol.secure_aggregate(
            client_updates=client_updates,
            dropout_clients=dropout_clients
        )

        # 結果の検証
        assert aggregated_result is not None, "ドロップアウトありでも集約が成功すること"
        assert len(aggregated_result) == 2, "集約結果の構造が正しいこと"

        # 期待値の計算（参加した3クライアントの平均）
        expected_layer1 = np.array([1.0, 2.0]) * 3 / 3  # [1.0, 2.0]
        expected_layer2 = np.array([3.0, 4.0]) * 3 / 3  # [3.0, 4.0]

        np.testing.assert_array_almost_equal(aggregated_result[0], expected_layer1)
        np.testing.assert_array_almost_equal(aggregated_result[1], expected_layer2)

    def test_secure_aggregation_insufficient_clients(self):
        """不十分なクライアント数での集約テスト"""
        from backend.federated_learning.security.secure_aggregation_protocol import SecureAggregationProtocol

        protocol = SecureAggregationProtocol(
            total_clients=5,
            min_clients=3,
            key_size=1024
        )

        # 最小クライアント数を下回る場合
        client_updates = {
            0: [np.array([1.0, 2.0])],
            1: [np.array([2.0, 3.0])]
        }
        dropout_clients = [2, 3, 4]  # 3クライアントがドロップアウト

        # 例外が発生することを期待
        with pytest.raises(ValueError, match="不十分なクライアント数"):
            protocol.secure_aggregate(
                client_updates=client_updates,
                dropout_clients=dropout_clients
            )

    def test_malicious_client_detection(self):
        """悪意のあるクライアント検出のテスト

        要件:
        - 異常に大きな更新値を送信するクライアントを検出
        - コミット・リビール方式での不正検出
        - ゼロ知識証明による検証
        """
        from backend.federated_learning.security.secure_aggregation_protocol import SecureAggregationProtocol

        protocol = SecureAggregationProtocol(
            total_clients=4,
            min_clients=3,
            key_size=1024,
            enable_malicious_detection=True
        )

        # 正常なクライアントの更新
        normal_updates = {
            0: [np.array([1.0, 2.0])],
            1: [np.array([1.1, 2.1])],
            2: [np.array([0.9, 1.9])]
        }

        # 悪意のあるクライアントの更新（異常に大きな値）
        malicious_updates = {
            3: [np.array([1000.0, 1000.0])]  # 異常に大きな値
        }

        all_updates = {**normal_updates, **malicious_updates}

        # 悪意のあるクライアントが検出されることを期待
        with pytest.raises(ValueError, match="悪意のあるクライアントを検出"):
            protocol.secure_aggregate(client_updates=all_updates)

        # 検出されたクライアントIDを確認
        detected_malicious = protocol.get_detected_malicious_clients()
        assert 3 in detected_malicious, "悪意のあるクライアント3が検出されること"

    def test_aggregation_correctness(self):
        """集約の正確性テスト

        要件:
        - セキュア集約の結果が平文集約と同じであること
        - 暗号化された状態で集約が実行されること
        - 最終的な復号化が正確であること
        """
        from backend.federated_learning.security.secure_aggregation_protocol import SecureAggregationProtocol

        protocol = SecureAggregationProtocol(
            total_clients=3,
            min_clients=3,
            key_size=1024
        )

        # テスト用の更新データ
        client_updates = {
            0: [np.array([1.0, 2.0]), np.array([5.0])],
            1: [np.array([3.0, 4.0]), np.array([6.0])],
            2: [np.array([5.0, 6.0]), np.array([7.0])]
        }

        # セキュア集約を実行
        secure_result = protocol.secure_aggregate(client_updates=client_updates)

        # 平文での集約（比較用）
        plaintext_result = protocol.plaintext_aggregate(client_updates=client_updates)

        # セキュア集約と平文集約の結果が一致することを確認
        assert len(secure_result) == len(plaintext_result)

        for i, (secure_layer, plaintext_layer) in enumerate(zip(secure_result, plaintext_result)):
            np.testing.assert_array_almost_equal(
                secure_layer, plaintext_layer,
                err_msg=f"レイヤー{i}の集約結果が一致しません"
            )

    def test_masking_mechanism(self):
        """マスキング機構のテスト

        要件:
        - 各クライアントが他のクライアント用のマスクを生成
        - マスクの合計がゼロになること
        - マスクが暗号化されて送信されること
        """
        from backend.federated_learning.security.secure_aggregation_protocol import SecureAggregationProtocol

        protocol = SecureAggregationProtocol(
            total_clients=3,
            min_clients=3,
            key_size=1024
        )

        # クライアント0のマスク生成
        masks = protocol.generate_client_masks(
            client_id=0,
            other_client_ids=[1, 2],
            model_shape=[(2,), (1,)]  # 2次元と1次元のレイヤー
        )

        # マスクが生成されることを確認
        assert isinstance(masks, dict), "マスクが辞書形式で生成されること"
        assert 1 in masks and 2 in masks, "他のクライアント用のマスクが生成されること"

        # マスクの形状確認
        for client_id in [1, 2]:
            mask = masks[client_id]
            assert len(mask) == 2, "モデル形状と一致するマスクが生成されること"
            assert mask[0].shape == (2,), "第1レイヤーのマスク形状が正しいこと"
            assert mask[1].shape == (1,), "第2レイヤーのマスク形状が正しいこと"

    def test_secret_sharing(self):
        """シークレット共有のテスト

        要件:
        - 鍵が複数の共有に分割されること
        - 閾値以上の共有があれば鍵を復元できること
        - 閾値未満では鍵を復元できないこと
        """
        from backend.federated_learning.security.secure_aggregation_protocol import SecureAggregationProtocol

        protocol = SecureAggregationProtocol(
            total_clients=5,
            min_clients=3,
            key_size=1024
        )

        # シークレット共有の実行（16バイト以下の秘密を使用）
        secret = b"test_secret_key!"  # 16バイトのテスト秘密
        shares = protocol.create_secret_shares(
            secret=secret,
            total_shares=5,
            threshold=3
        )

        # 共有数の確認
        assert len(shares) == 5, "指定した数の共有が生成されること"

        # 閾値以上の共有で復元成功
        selected_shares = {i: shares[i] for i in [0, 1, 2]}  # 3つの共有
        recovered_secret = protocol.recover_secret_from_shares(selected_shares, threshold=3)
        assert recovered_secret == secret, "閾値以上の共有で秘密を復元できること"

        # 閾値未満では復元失敗
        insufficient_shares = {i: shares[i] for i in [0, 1]}  # 2つの共有
        with pytest.raises(ValueError, match="不十分な共有数"):
            protocol.recover_secret_from_shares(insufficient_shares, threshold=3)

    def test_zero_knowledge_proof(self):
        """ゼロ知識証明のテスト

        要件:
        - クライアントが更新の妥当性を証明できること
        - 実際の更新値を明かすことなく証明できること
        - 不正な証明は検証に失敗すること
        """
        from backend.federated_learning.security.secure_aggregation_protocol import SecureAggregationProtocol

        protocol = SecureAggregationProtocol(
            total_clients=3,
            min_clients=3,
            key_size=1024,
            enable_zero_knowledge_proof=True
        )

        # 正当な更新の証明生成
        valid_update = [np.array([1.0, 2.0])]
        proof = protocol.generate_zero_knowledge_proof(
            client_id=0,
            model_update=valid_update,
            bound=10.0  # 更新値の上限
        )

        # 証明の検証
        is_valid = protocol.verify_zero_knowledge_proof(
            client_id=0,
            proof=proof,
            bound=10.0
        )
        assert is_valid, "正当な更新の証明が検証されること"

        # 不正な証明の検証失敗
        invalid_update = [np.array([100.0, 200.0])]  # 上限を超える値
        invalid_proof = protocol.generate_zero_knowledge_proof(
            client_id=1,
            model_update=invalid_update,
            bound=10.0
        )

        is_invalid = protocol.verify_zero_knowledge_proof(
            client_id=1,
            proof=invalid_proof,
            bound=10.0
        )
        assert not is_invalid, "不正な更新の証明が検証に失敗すること"

    @pytest.mark.asyncio
    async def test_async_secure_aggregation(self):
        """非同期セキュア集約のテスト

        要件:
        - 複数クライアントから並行して更新を受信
        - 非同期でセキュア集約を実行
        - タイムアウト処理
        """
        from backend.federated_learning.security.secure_aggregation_protocol import SecureAggregationProtocol

        protocol = SecureAggregationProtocol(
            total_clients=3,
            min_clients=3,
            key_size=1024
        )

        # 非同期でクライアント更新をシミュレート
        async def client_update_simulation(client_id: int, delay: float):
            await asyncio.sleep(delay)
            return {
                'client_id': client_id,
                'update': [np.array([client_id * 1.0, client_id * 2.0])]
            }

        # 並行してクライアント更新を収集
        tasks = [
            client_update_simulation(0, 0.1),
            client_update_simulation(1, 0.2),
            client_update_simulation(2, 0.3)
        ]

        client_results = await asyncio.gather(*tasks)

        # 更新をプロトコルに渡す
        client_updates = {}
        for result in client_results:
            client_updates[result['client_id']] = result['update']

        # 非同期セキュア集約を実行
        aggregated_result = await protocol.async_secure_aggregate(
            client_updates=client_updates,
            timeout=5.0
        )

        # 結果の検証
        assert aggregated_result is not None, "非同期セキュア集約が成功すること"
        assert len(aggregated_result) == 1, "集約結果の構造が正しいこと"

    def test_performance_benchmark(self):
        """セキュア集約のパフォーマンステスト

        要件（TDD.yamlより）:
        - エラーハンドリング強化
        - 並列処理での効率化
        """
        from backend.federated_learning.security.secure_aggregation_protocol import SecureAggregationProtocol
        import time

        protocol = SecureAggregationProtocol(
            total_clients=3,  # テスト用に小さなクライアント数
            min_clients=3,
            key_size=1024  # テスト用に小さなキーサイズ
        )

        # 小さめのモデル更新を準備（Paillier暗号は計算コストが高いため）
        small_updates = {}
        for client_id in range(3):
            # 各レイヤーの更新（テスト用に小さなサイズ）
            updates = [
                np.random.normal(0, 0.1, (5, 3)),    # 小さな隠れ層
                np.random.normal(0, 0.1, (3, 2)),    # 小さな出力層
                np.random.normal(0, 0.1, (2,))       # バイアス
            ]
            small_updates[client_id] = updates

        # パフォーマンス測定
        start_time = time.time()
        aggregated_result = protocol.secure_aggregate(client_updates=small_updates)
        aggregation_time = time.time() - start_time

        # Paillier暗号を考慮した現実的な時間制限（5分）
        assert aggregation_time < 300.0, f"セキュア集約が遅すぎます: {aggregation_time}秒"
        assert aggregated_result is not None, "集約結果が正しく生成されること"
        assert len(aggregated_result) == 3, "全レイヤーが正しく集約されること"

@pytest.fixture
def mock_secure_aggregation_protocol():
    """テスト用のモックSecureAggregationProtocolを提供"""
    with patch('backend.federated_learning.security.secure_aggregation_protocol.SecureAggregationProtocol') as mock:
        instance = Mock()
        mock.return_value = instance

        # デフォルトの動作を設定
        instance.secure_aggregate.return_value = [np.array([1.0, 2.0]), np.array([3.0])]
        instance.plaintext_aggregate.return_value = [np.array([1.0, 2.0]), np.array([3.0])]
        instance.get_detected_malicious_clients.return_value = []
        instance.generate_client_masks.return_value = {1: [np.array([0.1, 0.2])], 2: [np.array([-0.1, -0.2])]}
        instance.create_secret_shares.return_value = {i: f"share_{i}" for i in range(5)}
        instance.recover_secret_from_shares.return_value = b"recovered_secret"
        instance.generate_zero_knowledge_proof.return_value = "valid_proof"
        instance.verify_zero_knowledge_proof.return_value = True

        # 非同期メソッドのモック
        instance.async_secure_aggregate = AsyncMock(return_value=[np.array([1.0, 2.0])])

        yield instance