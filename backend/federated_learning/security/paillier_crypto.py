"""
Paillier暗号実装

TDD Phase 1, Task 1.1: Paillier暗号ライブラリの統合
GREEN段階: テストを通す最小限のコードを実装
"""

import logging
import json
import base64
from typing import Union, Any
from phe import paillier
import pickle

logger = logging.getLogger(__name__)

class PaillierCrypto:
    """Paillier暗号の実装

    python-paillierライブラリを使用したPaillier暗号の実装。
    準同型加算、暗号化/復号化、シリアライゼーション機能を提供。
    """

    def __init__(self, key_size: int = 2048):
        """初期化

        Args:
            key_size: 鍵のサイズ（ビット）、デフォルトは2048
        """
        self.key_size = key_size

        # 鍵ペアを生成
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_size)

        logger.info(f"Paillier暗号初期化完了: key_size={key_size}")

    def encrypt(self, plaintext: Union[int, float]) -> paillier.EncryptedNumber:
        """数値を暗号化

        Args:
            plaintext: 暗号化する平文（数値）

        Returns:
            暗号化された数値
        """
        try:
            encrypted = self.public_key.encrypt(plaintext)
            logger.debug(f"暗号化完了: {plaintext} -> [暗号化済み]")
            return encrypted
        except Exception as e:
            logger.error(f"暗号化エラー: {e}")
            raise

    def decrypt(self, ciphertext: paillier.EncryptedNumber) -> Union[int, float]:
        """暗号化された数値を復号化

        Args:
            ciphertext: 暗号化された数値

        Returns:
            復号化された平文
        """
        try:
            decrypted = self.private_key.decrypt(ciphertext)
            logger.debug(f"復号化完了: [暗号化済み] -> {decrypted}")
            return decrypted
        except Exception as e:
            logger.error(f"復号化エラー: {e}")
            raise

    def add_encrypted(self,
                     encrypted_a: paillier.EncryptedNumber,
                     encrypted_b: paillier.EncryptedNumber) -> paillier.EncryptedNumber:
        """暗号化された数値同士の準同型加算

        Args:
            encrypted_a: 暗号化された数値A
            encrypted_b: 暗号化された数値B

        Returns:
            暗号化されたままの加算結果
        """
        try:
            result = encrypted_a + encrypted_b
            logger.debug("準同型加算完了")
            return result
        except Exception as e:
            logger.error(f"準同型加算エラー: {e}")
            raise

    def serialize_encrypted(self, encrypted: paillier.EncryptedNumber) -> str:
        """暗号化されたデータをシリアライズ

        Args:
            encrypted: 暗号化されたデータ

        Returns:
            シリアライズされた文字列
        """
        try:
            # EncryptedNumberを辞書形式に変換
            data = {
                'ciphertext': encrypted.ciphertext(),
                'exponent': encrypted.exponent,
                'public_key_n': encrypted.public_key.n
            }

            # JSON文字列に変換
            serialized = json.dumps(data)
            logger.debug("暗号化データのシリアライズ完了")
            return serialized
        except Exception as e:
            logger.error(f"シリアライズエラー: {e}")
            raise

    def deserialize_encrypted(self, serialized: str) -> paillier.EncryptedNumber:
        """シリアライズされたデータを暗号化データに復元

        Args:
            serialized: シリアライズされた文字列

        Returns:
            暗号化されたデータ
        """
        try:
            # JSON文字列を辞書に変換
            data = json.loads(serialized)

            # 公開鍵を復元（既存の公開鍵と一致する必要がある）
            if data['public_key_n'] != self.public_key.n:
                raise ValueError("公開鍵が一致しません")

            # EncryptedNumberを復元
            encrypted = paillier.EncryptedNumber(
                self.public_key,
                data['ciphertext'],
                data['exponent']
            )

            logger.debug("暗号化データのデシリアライズ完了")
            return encrypted
        except Exception as e:
            logger.error(f"デシリアライズエラー: {e}")
            raise

    def get_public_key_dict(self) -> dict:
        """公開鍵を辞書形式で取得

        クライアント間での公開鍵共有に使用

        Returns:
            公開鍵の辞書表現
        """
        return {
            'n': self.public_key.n,
            'g': self.public_key.g,
            'max_int': self.public_key.max_int
        }

    def load_public_key_from_dict(self, key_dict: dict) -> paillier.PaillierPublicKey:
        """辞書から公開鍵を復元

        Args:
            key_dict: 公開鍵の辞書表現

        Returns:
            復元された公開鍵
        """
        return paillier.PaillierPublicKey(key_dict['n'])

    def __str__(self) -> str:
        """文字列表現"""
        return f"PaillierCrypto(key_size={self.key_size}, public_key_n={self.public_key.n})"

    def __repr__(self) -> str:
        """デバッグ用文字列表現"""
        return self.__str__()