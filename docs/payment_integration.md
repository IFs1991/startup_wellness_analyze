# Pay.jp 決済連携の実装とセキュリティ

## 概要

本システムでは、サブスクリプション課金のために [Pay.jp](https://pay.jp/) を利用しています。
このドキュメントでは、バックエンドにおけるPay.jp連携の実装方法と、関連するセキュリティ対策について説明します。

## 実装内容

### 1. コアモジュール (`core/subscription_manager.py`)

Pay.jp APIとの連携ロジックを集約した `SubscriptionManager` クラスを実装しました。

- **主な機能:**
    - **顧客管理:** `get_or_create_customer`: Pay.jpの顧客オブジェクトを取得または新規作成します。Firestoreのユーザーデータと `payjp_customer_id` で紐付けます。
    - **プラン取得:** `list_plans`: Pay.jpに登録されているプランの一覧を取得します。
    - **サブスクリプション作成/更新:** `create_or_update_subscription`: 新規サブスクリプションの作成、または既存サブスクリプションのプラン変更を行います。
    - **サブスクリプションキャンセル:** `cancel_subscription`: サブスクリプションをキャンセルします（期間終了時キャンセル/即時キャンセル）。
    - **状態取得:** `get_subscription_details`: FirestoreとPay.jpからユーザーの最新のサブスクリプション情報を取得します。
    - **Webhook処理:** `handle_webhook`: Pay.jpからのWebhookイベント（支払い成功/失敗、サブスクリプション変更など）を検証し、Firestoreのデータを更新します。
- **依存関係:**
    - `FirestoreService`: Firestoreとのデータ永続化のため。
    - `AuthManager`: （将来的に）ユーザーEmailなどの情報を取得するため（現在はモック）。
- **設定:**
    - 環境変数 `PAYJP_SECRET_KEY` と `PAYJP_WEBHOOK_SECRET` が必要です。

### 2. APIエンドポイント (`main.py`)

サブスクリプション関連の操作を行うためのHTTP APIエンドポイントを `/api/subscription/` 以下に実装しました。

- `/api/subscription/plans` (GET): 利用可能なプラン一覧を返します。
- `/api/subscription/status` (GET): 認証ユーザーの現在のサブスクリプション状態を返します。
- `/api/subscription/change` (POST): 認証ユーザーのサブスクリプションを作成または変更します。
    - Request Body: `{ "plan_id": "pln_xxxx", "payment_method_id": "pm_xxxx" }` (payment_method_idは初回登録時などに必要)
- `/api/subscription/cancel` (POST): 認証ユーザーのサブスクリプションをキャンセルします。
    - Request Body: `{ "at_period_end": true }` (省略可能、デフォルトtrue)

### 3. Webhookエンドポイント (`main.py`)

Pay.jpからの非同期通知を受け取るためのエンドポイント `/webhooks/payjp` (POST) を実装しました。

- Pay.jpからのリクエストヘッダー `Payjp-Signature` を使用してリクエストの正当性を検証します。
- `SubscriptionManager.handle_webhook` を呼び出し、イベントタイプに応じた処理（主にFirestoreのステータス更新）を実行します。

### 4. 認証 (`main.py`)

サブスクリプションAPIエンドポイントは、Firebase IDトークンによる認証を必須とします。

- `Authorization: Bearer <Firebase ID Token>` ヘッダーが必要です。
- `get_current_firebase_user` 依存性注入関数がトークンを検証し、ユーザー情報を取得します。

## セキュリティ対策と考慮事項

### 1. APIキー管理

- **対策:** Pay.jpの秘密鍵 (`PAYJP_SECRET_KEY`) およびWebhook検証鍵 (`PAYJP_WEBHOOK_SECRET`) は、環境変数から読み込むように実装されており、コード中に直接記述されていません。
- **考慮事項:** 本番環境では、環境変数を安全な方法（例: クラウドプロバイダーのシークレット管理サービス、`.env` ファイルの適切な管理）で設定・管理する必要があります。

### 2. Webhook検証

- **対策:** `/webhooks/payjp` エンドポイントは、受信したリクエストの `Payjp-Signature` ヘッダーと `PAYJP_WEBHOOK_SECRET` を用いて署名を検証します。これにより、Pay.jp以外からの不正なリクエストを拒否します。
- **考慮事項:** `PAYJP_WEBHOOK_SECRET` が環境変数に正しく設定されていることを確認してください。設定されていない場合、検証がスキップされ、セキュリティリスクとなります（現在は警告ログのみ）。

### 3. 通信の暗号化

- **対策:** Pay.jp APIとの通信は、`payjp` Pythonライブラリによって自動的にHTTPSで行われます。
- **考慮事項:** Webhookを受信するサーバーもHTTPSで公開されている必要があります（例: ロードバランサやリバースプロキシでSSL終端）。

### 4. 認証と認可

- **対策:** サブスクリプション関連APIはFirebase IDトークンによる認証を必須とし、リクエストを行ったユーザー本人のみが自身のサブスクリプションを操作できるようにしています (`get_current_firebase_user` 依存関係）。
- **考慮事項:**
    - `get_current_firebase_user` 関数が正しくFirebase Admin SDKと連携していることを確認してください。
    - 今後、管理者用の操作（例: 全ユーザーのサブスクリプション一覧）を実装する場合は、別途ロールベースの認可制御が必要です。

### 5. 決済情報の取り扱い (PCI DSS)

- **対策:** バックエンドでは、**クレジットカード番号などの生の決済情報を扱わない** 設計になっています。フロントエンドがPay.jp ElementsやCheckoutなどのセキュアな方法でカード情報をトークン化（または支払い方法ID化）し、そのIDのみをバックエンドに送信することを前提としています。
- **考慮事項:** フロントエンドの実装がPay.jpの推奨する方法に従っていることを確認してください。サーバーサイドでカード情報を保持・処理することは絶対に避けてください。

### 6. Firestoreセキュリティルール

- **対策:** Firestoreに保存されるユーザーごとのサブスクリプション情報 (`payjp_customer_id`, `payjp_subscription_id`など) は、Firestoreセキュリティルールによって保護されるべきです。
- **考慮事項:** ユーザーが自身の情報にのみアクセス・更新できるように、適切なセキュリティルールを設定してください。特にクライアントSDKからFirestoreに直接アクセスする場合は重要です。

### 7. エラーハンドリング

- **対策:** APIエンドポイントやWebhookハンドラーでは、エラー発生時に機密情報（詳細なPay.jpエラーメッセージなど）がクライアントに漏洩しないように、一般的なエラーメッセージを返すように実装されています。
- **考慮事項:** 詳細なエラー情報はサーバーログに記録し、デバッグや監視に活用してください。

## 今後の課題

- 請求書発行機能の実装
- 使用量ベース課金ロジックの検討・実装
- 管理者向け機能（ユーザー検索、手動でのサブスクリプション操作など）の実装と認可制御
- フロントエンドにおけるPay.jp Elements/Checkoutの具体的な連携方法のドキュメント化