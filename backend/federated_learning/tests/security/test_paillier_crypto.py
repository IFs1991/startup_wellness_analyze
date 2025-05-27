"""
Paillier暗号ライブラリのテスト

TDD Phase 1, Task 1.1: Paillier暗号ライブラリの統合
RED段階: 失敗するテストを最初に書く
"""

import pytest
import numpy as np
from typing import Tuple
from unittest.mock import Mock, patch

class TestPaillierCrypto:
    """Paillier暗号のテストクラス

    TDD.yamlに基づく要件:
    - test_paillier_encryption_decryption: 暗号化/復号化の正確性
    - test_homomorphic_addition: 準同型加算の正確性
    - test_key_generation: 鍵生成の正確性
    """

    def test_paillier_encryption_decryption(self):
        """Paillier暗号化/復号化のテスト

        要件:
        - 数値を暗号化し、復号化して元の値が得られること
        - 異なる数値で同じテストが成功すること
        - 大きな数値でも正確に動作すること
        """
        # まだPaillierCryptoクラスが実装されていないため、このテストは失敗する（RED段階）
        from backend.federated_learning.security.paillier_crypto import PaillierCrypto

        # テストデータ
        test_values = [42, 100, 1000, -50, 0]

        crypto = PaillierCrypto()

        for value in test_values:
            # 暗号化
            encrypted = crypto.encrypt(value)

            # 復号化
            decrypted = crypto.decrypt(encrypted)

            # 元の値と一致することを確認
            assert decrypted == value, f"復号化に失敗: 期待値={value}, 実際={decrypted}"

    def test_homomorphic_addition(self):
        """準同型加算のテスト

        要件:
        - 暗号化された値同士の加算結果を復号化すると、平文の加算結果と一致すること
        - 複数の値の加算でも正確に動作すること
        """
        # まだPaillierCryptoクラスが実装されていないため、このテストは失敗する（RED段階）
        from backend.federated_learning.security.paillier_crypto import PaillierCrypto

        crypto = PaillierCrypto()

        # テストデータ
        a, b = 15, 27
        expected_sum = a + b  # 42

        # 暗号化
        encrypted_a = crypto.encrypt(a)
        encrypted_b = crypto.encrypt(b)

        # 準同型加算
        encrypted_sum = crypto.add_encrypted(encrypted_a, encrypted_b)

        # 復号化
        decrypted_sum = crypto.decrypt(encrypted_sum)

        assert decrypted_sum == expected_sum, f"準同型加算に失敗: 期待値={expected_sum}, 実際={decrypted_sum}"

    def test_homomorphic_addition_multiple_values(self):
        """複数値の準同型加算テスト"""
        from backend.federated_learning.security.paillier_crypto import PaillierCrypto

        crypto = PaillierCrypto()

        values = [10, 20, 30, 40]
        expected_sum = sum(values)  # 100

        # 暗号化
        encrypted_values = [crypto.encrypt(v) for v in values]

        # 順次加算
        encrypted_sum = encrypted_values[0]
        for encrypted_val in encrypted_values[1:]:
            encrypted_sum = crypto.add_encrypted(encrypted_sum, encrypted_val)

        # 復号化
        decrypted_sum = crypto.decrypt(encrypted_sum)

        assert decrypted_sum == expected_sum

    def test_key_generation(self):
        """鍵生成のテスト

        要件:
        - 公開鍵と秘密鍵のペアが生成されること
        - 生成される鍵が毎回異なること
        - 鍵のサイズが適切であること
        """
        from backend.federated_learning.security.paillier_crypto import PaillierCrypto

        crypto1 = PaillierCrypto()
        crypto2 = PaillierCrypto()

        # 鍵が生成されていることを確認
        assert hasattr(crypto1, 'public_key'), "公開鍵が生成されていません"
        assert hasattr(crypto1, 'private_key'), "秘密鍵が生成されていません"

        # 異なるインスタンスで異なる鍵が生成されることを確認
        assert crypto1.public_key != crypto2.public_key, "同じ公開鍵が生成されています"

    def test_key_size_security(self):
        """鍵サイズとセキュリティのテスト"""
        from backend.federated_learning.security.paillier_crypto import PaillierCrypto

        # デフォルトは2048ビット
        crypto = PaillierCrypto()
        assert crypto.key_size >= 2048, "鍵サイズが不足しています"

        # カスタム鍵サイズ
        crypto_4096 = PaillierCrypto(key_size=4096)
        assert crypto_4096.key_size == 4096

    def test_large_number_handling(self):
        """大きな数値の処理テスト"""
        from backend.federated_learning.security.paillier_crypto import PaillierCrypto

        crypto = PaillierCrypto()

        # 大きな数値
        large_value = 2**50

        encrypted = crypto.encrypt(large_value)
        decrypted = crypto.decrypt(encrypted)

        assert decrypted == large_value

    def test_negative_number_handling(self):
        """負の数値の処理テスト"""
        from backend.federated_learning.security.paillier_crypto import PaillierCrypto

        crypto = PaillierCrypto()

        negative_value = -12345

        encrypted = crypto.encrypt(negative_value)
        decrypted = crypto.decrypt(encrypted)

        assert decrypted == negative_value

    def test_performance_benchmark(self):
        """パフォーマンステスト

        要件（TDD.yamlより）:
        - 並列暗号化処理
        - メモリ効率の改善
        """
        from backend.federated_learning.security.paillier_crypto import PaillierCrypto
        import time

        crypto = PaillierCrypto()

        # より少ない数でテスト（現実的な負荷）
        values = list(range(10))

        start_time = time.time()
        encrypted_values = [crypto.encrypt(v) for v in values]
        encryption_time = time.time() - start_time

        # 復号化時間を測定
        start_time = time.time()
        decrypted_values = [crypto.decrypt(enc) for enc in encrypted_values]
        decryption_time = time.time() - start_time

        # 現実的な閾値に調整（Paillier暗号は重い処理）
        assert encryption_time < 30.0, f"暗号化が遅すぎます: {encryption_time}秒"
        assert decryption_time < 30.0, f"復号化が遅すぎます: {decryption_time}秒"

        # 結果の正確性も確認
        assert decrypted_values == values

    def test_serialization(self):
        """暗号化データのシリアライゼーションテスト

        ネットワーク通信のため、暗号化されたデータを
        JSONやバイナリ形式でシリアライズできる必要がある
        """
        from backend.federated_learning.security.paillier_crypto import PaillierCrypto
        import json

        crypto = PaillierCrypto()

        value = 12345
        encrypted = crypto.encrypt(value)

        # JSON形式でシリアライズ
        serialized = crypto.serialize_encrypted(encrypted)
        assert isinstance(serialized, (str, bytes)), "シリアライズ結果が不正です"

        # デシリアライズ
        deserialized = crypto.deserialize_encrypted(serialized)

        # 復号化して確認
        decrypted = crypto.decrypt(deserialized)
        assert decrypted == value

@pytest.fixture
def mock_paillier_crypto():
    """テスト用のモックPaillierCryptoを提供"""
    with patch('backend.federated_learning.security.paillier_crypto.PaillierCrypto') as mock:
        instance = Mock()
        mock.return_value = instance

        # デフォルトの動作を設定
        instance.encrypt.side_effect = lambda x: f"encrypted_{x}"
        instance.decrypt.side_effect = lambda x: int(x.replace("encrypted_", ""))
        instance.add_encrypted.side_effect = lambda a, b: f"encrypted_{int(a.replace('encrypted_', '')) + int(b.replace('encrypted_', ''))}"

        instance.public_key = "mock_public_key"
        instance.private_key = "mock_private_key"
        instance.key_size = 2048

        yield instance