# Firestore データベース設計指針

## 1. 概要

本プロジェクトでは、データストレージとしてCloud Firestoreを採用し、一本化します。この文書では、その設計指針と実装方針について説明します。

## 2. 採用理由

### 2.1 技術的メリット
- リアルタイムデータ同期機能
- スケーラビリティの高さ
- GCPエコシステムとの親和性
- 運用コストの最適化
- インフラ管理の簡素化

### 2.2 ビジネス的メリット
- 開発速度の向上
- 保守性の向上
- インフラコストの予測可能性
- 運用工数の削減

## 3. データモデル設計

### 3.1 コレクション構造
```javascript
{
  "users": {
    // ユーザー情報
    "[uid]": {
      "profile": {},
      "settings": {},
      "metadata": {}
    }
  },
  "consultations": {
    // 問診データ
    "[consultation_id]": {
      "user_id": "",
      "timestamp": "",
      "data": {}
    }
  },
  "treatments": {
    // 治療データ
    "[treatment_id]": {
      "user_id": "",
      "consultation_id": "",
      "data": {}
    }
  },
  "analytics": {
    // 分析結果
    "[analysis_id]": {
      "type": "",
      "results": {},
      "metadata": {}
    }
  }
}
```

### 3.2 インデックス設計
- ユーザーID + タイムスタンプ
- 問診ID + 治療ID
- 分析タイプ + タイムスタンプ

## 4. データアクセスパターン

### 4.1 基本的なアクセスパターン
```python
# シングルドキュメント取得
doc = db.collection('users').document(user_id).get()

# コレクションクエリ
docs = db.collection('consultations')\
         .where('user_id', '==', user_id)\
         .order_by('timestamp', direction='DESC')\
         .limit(10)\
         .stream()
```

### 4.2 バッチ処理パターン
```python
# バッチ書き込み
batch = db.batch()
batch.set(doc_ref, data)
batch.commit()
```

## 5. 分析データ処理

### 5.1 データフロー
1. Firestoreからデータ取得
2. メモリ内で分析処理
3. 結果をFirestoreに保存
4. 必要に応じてCloud Storageに中間データ保存

### 5.2 分析モジュール統合
- 記述統計（calculate_descriptive_stats.py）
- 相関分析（correlation_analysis.py）
- 時系列分析（TimeSeriesAnalyzer.py）
- アソシエーション分析（AssociationAnalyzer.py）
- クラスタリング（ClusterAnalyzer.py）
- 主成分分析（PCAAnalyzer.py）
- 生存分析（SurvivalAnalyzer.py）
- テキストマイニング（TextMiner.py）

## 6. パフォーマンス最適化

### 6.1 クエリ最適化
- 適切なインデックス設計
- クエリの効率化
- ページネーションの実装

### 6.2 キャッシュ戦略
- Cloud Storageを使用した中間データのキャッシュ
- メモリキャッシュの活用
- クライアントサイドキャッシュの実装

## 7. セキュリティ設計

### 7.1 認証
- Firebase Authentication利用
- JWTトークンによる認証

### 7.2 アクセス制御
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId} {
      allow read: if request.auth.uid == userId;
      allow write: if request.auth.uid == userId;
    }
    // その他のセキュリティルール
  }
}
```

## 8. 監視とメンテナンス

### 8.1 モニタリング項目
- クエリパフォーマンス
- データ容量
- エラーレート
- レイテンシ

### 8.2 バックアップ戦略
- 定期的なエクスポート
- Point-in-time リカバリ
- 災害復旧計画

## 9. 移行計画

### 9.1 PostgreSQLからの移行
1. 既存の設定の削除
2. 環境変数の更新
3. データアクセス層の一本化
4. 動作確認とテスト

## 10. 今後の展望

### 10.1 スケーリング戦略
- シャーディングの検討
- 読み取り/書き込みの最適化
- キャッシュ戦略の進化

### 10.2 機能拡張
- BigQueryとの連携
- AIモデルとの統合
- リアルタイム分析の強化