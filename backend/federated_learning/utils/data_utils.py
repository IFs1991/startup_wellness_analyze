"""
データ処理ユーティリティ

このモジュールは、連合学習システムのデータ処理ユーティリティ関数を提供します。
データの前処理や分割などの機能が含まれます。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def preprocess_data(data: Union[pd.DataFrame, np.ndarray],
                    categorical_cols: Optional[List[str]] = None,
                    numerical_cols: Optional[List[str]] = None,
                    scaler_type: str = "standard") -> Tuple[np.ndarray, Dict[str, Any]]:
    """データを前処理する

    Args:
        data: 前処理するデータ
        categorical_cols: カテゴリカル列のリスト
        numerical_cols: 数値列のリスト
        scaler_type: スケーラーのタイプ ("standard" または "minmax")

    Returns:
        前処理されたデータと前処理パラメータのタプル
    """
    logger.info("データの前処理を開始します")

    # DataFrameをnumpy配列に変換
    if isinstance(data, pd.DataFrame):
        # カテゴリカル列の処理
        if categorical_cols is not None:
            # ワンホットエンコーディング
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # 数値列の選択
        if numerical_cols is not None:
            # 数値列を選択
            data_array = data[numerical_cols].values
        else:
            # 全ての列を使用
            data_array = data.values
    else:
        # すでにnumpy配列の場合
        data_array = data

    # スケーリング
    preprocessing_params = {}
    if scaler_type == "standard":
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_array)
        preprocessing_params["scaler"] = scaler
        preprocessing_params["scaler_type"] = "standard"
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_array)
        preprocessing_params["scaler"] = scaler
        preprocessing_params["scaler_type"] = "minmax"
    else:
        # スケーリングなし
        data_scaled = data_array
        preprocessing_params["scaler_type"] = "none"

    # NANの処理
    if np.isnan(data_scaled).any():
        logger.warning("データに欠損値が含まれています。0で置換します。")
        data_scaled = np.nan_to_num(data_scaled)

    # データの形状を記録
    preprocessing_params["data_shape"] = data_scaled.shape

    logger.info(f"データの前処理が完了しました: 形状={data_scaled.shape}")

    return data_scaled, preprocessing_params

def split_data(X: np.ndarray,
              y: np.ndarray,
              test_size: float = 0.2,
              validation_size: float = 0.1,
              random_state: int = 42) -> Dict[str, np.ndarray]:
    """データを訓練、検証、テストセットに分割する

    Args:
        X: 特徴量データ
        y: ターゲットデータ
        test_size: テストセットの割合
        validation_size: 検証セットの割合
        random_state: 乱数シード

    Returns:
        分割されたデータセット
    """
    logger.info("データを分割しています...")

    # まず、訓練+検証セットとテストセットに分割
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 次に、訓練セットと検証セットに分割
    # 元のデータサイズに対する相対的な検証セットサイズを計算
    relative_validation_size = validation_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_validation_size, random_state=random_state
    )

    logger.info(f"データ分割が完了しました: 訓練={X_train.shape}, 検証={X_val.shape}, テスト={X_test.shape}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

def create_client_data_partitions(X: np.ndarray,
                                y: np.ndarray,
                                num_clients: int,
                                iid: bool = True,
                                alpha: float = 1.0,
                                random_state: int = 42) -> List[Dict[str, np.ndarray]]:
    """クライアント間でデータを分割する

    Args:
        X: 特徴量データ
        y: ターゲットデータ
        num_clients: クライアント数
        iid: IID（独立同一分布）分割の場合はTrue、非IIDの場合はFalse
        alpha: 非IID分割のディリクレ濃度パラメータ（小さいほど非IID性が強い）
        random_state: 乱数シード

    Returns:
        クライアントごとのデータ
    """
    logger.info(f"データを {num_clients} クライアントに分割しています: iid={iid}")

    np.random.seed(random_state)
    n_samples = X.shape[0]

    # クライアントごとのデータリスト
    client_data = []

    if iid:
        # IID分割: ランダムに均等にデータを分割
        indices = np.random.permutation(n_samples)
        batch_size = n_samples // num_clients

        for i in range(num_clients):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            client_indices = indices[start_idx:end_idx]
            client_data.append({
                "X": X[client_indices],
                "y": y[client_indices]
            })
    else:
        # 非IID分割: クラスごとにディリクレ分布に基づいて分割
        if len(y.shape) > 1 and y.shape[1] > 1:
            # one-hot エンコーディングの場合、クラスインデックスに変換
            y_idx = np.argmax(y, axis=1)
        else:
            y_idx = y.flatten()

        classes = np.unique(y_idx)
        num_classes = len(classes)

        # クラスごとにインデックスを取得
        class_indices = [np.where(y_idx == c)[0] for c in classes]

        # クライアントごとにクラス分布を生成（ディリクレ分布）
        client_class_distributions = np.random.dirichlet(
            alpha=alpha * np.ones(num_classes), size=num_clients
        )

        # 各クライアントのデータサイズを計算
        client_sample_sizes = [n_samples // num_clients] * num_clients
        remainder = n_samples % num_clients
        for i in range(remainder):
            client_sample_sizes[i] += 1

        # クライアントごとにデータを割り当て
        for i in range(num_clients):
            client_indices = []

            # このクライアントの必要サンプル数
            target_size = client_sample_sizes[i]

            # このクライアントのクラス分布
            class_probs = client_class_distributions[i]

            # クラスごとのサンプル数を計算
            class_samples = np.round(class_probs * target_size).astype(int)

            # 合計サンプル数を調整
            diff = target_size - np.sum(class_samples)
            if diff > 0:
                class_samples[np.argmax(class_probs)] += diff
            elif diff < 0:
                class_samples[np.argmin(class_probs)] += diff

            # 各クラスからサンプルを選択
            for c, n_samples in enumerate(class_samples):
                if n_samples > 0:
                    # このクラスから必要数のサンプルをランダムに選択
                    selected = np.random.choice(
                        class_indices[c],
                        size=min(n_samples, len(class_indices[c])),
                        replace=False
                    )
                    client_indices.extend(selected)

                    # 選択されたサンプルをクラスインデックスから削除（重複を避けるため）
                    class_indices[c] = np.setdiff1d(class_indices[c], selected)

            # このクライアントのデータを追加
            client_data.append({
                "X": X[client_indices],
                "y": y[client_indices]
            })

    # 各クライアントのデータサイズをログ
    for i, data in enumerate(client_data):
        logger.info(f"クライアント {i}: サンプル数={data['X'].shape[0]}")

    return client_data