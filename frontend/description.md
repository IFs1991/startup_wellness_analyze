スタートアップウェルネスアナライザー - 新フロントエンド構造
プロジェクト概要
スタートアップウェルネスアナライザーの新しいフロントエンドは、Next.js（App Router）を使用し、チャットモードと詳細分析モードを切り替え可能なモダンなUIを提供します。Tailwind CSSとshadcn/uiを活用した一貫性のあるデザインシステムを採用しています。
ディレクトリ構造
/app
Next.jsのApp Routerを使用したファイルベースのルーティングを提供します。
page.tsx: アプリケーションのエントリーポイント。AppShellコンポーネントをレンダリングします。
layout.tsx: すべてのページに適用されるレイアウト定義。
globals.css: グローバルスタイル定義。
/login/page.tsx: ログインページの実装。
/analysis: 詳細分析ページ関連のルート。
/companies: 企業管理ページ関連のルート。
/reports: レポート管理ページ関連のルート。
データフロー
一般的なデータ取得・表示のフローは以下のようになります。
バックエンドサーバー → /api (クライアントAPI層) → /hooks (データ取得・状態管理ロジック) → /app または /components (UI表示)
/components
再利用可能なUIコンポーネントを格納します。
コアコンポーネント
app-shell.tsx: アプリケーションの基本シェル。チャットモードと分析モードの切り替え機能を管理します。
view-switcher.tsx: チャットと分析の切り替えインターフェース。
header.tsx: アプリケーションヘッダー。
left-sidebar.tsx: 左サイドバー（企業リスト、フィルターなど）。
right-sidebar.tsx: 右サイドバー（詳細情報、コンテキスト情報など）。
モード関連コンポーネント
chat-view.tsx: チャットモード用のメインコンポーネント。メッセージの表示と入力を管理します。
analysis-view.tsx: 分析モード用のメインコンポーネント。左右のサイドバーと分析コンテンツを配置します。
analysis-content.tsx: 分析モードのメインコンテンツ領域。
analysis-layout.tsx: 分析モード用のレイアウト定義。
機能別コンポーネント
/analysis: 各種分析コンポーネント（時系列分析など）。
/charts: データ可視化用のグラフコンポーネント。
/companies: 企業データ管理関連コンポーネント。
/reports: レポート生成・管理関連コンポーネント。
/ui: 基本UIコンポーネント（shadcn/uiベース）。
/hooks
カスタムReactフックを提供します。
use-toast.ts: トースト通知用のフック。
use-mobile.tsx: モバイル表示検出用のフック。
**Note:** 既存の `@src/hooks` からコピーされたフックが含まれます。Next.js対応として、全フックに `"use client"` ディレクティブを追加し、以下のフックを特に重点的に修正しました：
- useAuth.ts: 認証情報へのアクセス（AuthContextとの連携）
- useWebSocketConnection.ts: WebSocket通信管理（ブラウザ環境チェック追加）
- useSubscription.ts: サブスクリプション状態管理

/lib
ユーティリティと定数を提供します。
utils.ts: 汎用ユーティリティ関数 (既存の `@src/lib/utils.ts` からコピー)。
constants.ts: アプリケーション全体で使用される定数。
ai-insights-generator.ts: AI関連機能 (既存の `@src/lib/ai-insights-generator.ts` からコピー)。依存関係の解決と動作確認が必要。
analysis-explanations.ts: 分析手法の説明データ (既存の `@src/data/analysis-explanations.ts` からコピー)。

/types
TypeScript型定義を提供します。
**Note:** 既存の `@src/types` からコピーされた型定義が含まれます。

/contexts
Reactコンテキストを提供します。
AuthContext.tsx: 認証状態管理（Firebase Authと連携）

/styles
スタイリングに関連するファイルを格納します。

/api
バックエンドAPIとの通信関連モジュールを格納します。
**Note:** 既存の `@src/api` からコピーされました。Next.js対応として `"use client"` ディレクティブの追加、環境変数参照の変更（`import.meta.env` → `process.env`）、ブラウザ環境チェック（`typeof window !== 'undefined'`）の追加を行いました。

/firebase
Firebase関連の設定と初期化コードを格納します。
**Note:** 既存の `@src/firebase` からコピーされました。Next.js対応として `"use client"` ディレクティブの追加、環境変数参照の変更（`import.meta.env.VITE_*` → `process.env.NEXT_PUBLIC_*`）を行いました。

その他の設定ファイル
tailwind.config.ts: Tailwind CSSの設定。
components.json: shadcn/uiの設定。
next.config.mjs: Next.jsの設定。
チャットモードと分析モードの実装
チャットモード
チャットモードは以下のファイルと関連しています：
components/chat-view.tsx: チャットモードのメインコンポーネント。
メッセージリストの表示
メッセージ入力フォーム
レスポンス処理
components/suggested-queries.tsx: 提案クエリ表示コンポーネント。
ユーザーが使用できる質問例の提案
分析モード
分析モードは以下のファイルと関連しています：
components/analysis-view.tsx: 分析モードのメインコンテナ。
左右のサイドバーを配置
中央の分析コンテンツ領域を管理
components/analysis-content.tsx: 分析データと可視化の表示。
components/analysis/*.tsx: 各種分析手法の個別コンポーネント。
例: time-series-analysis.tsx
components/charts/*.tsx: データ可視化用グラフコンポーネント。
モード切り替え
モード切り替えは以下のファイルで実装されています：
components/app-shell.tsx:
currentView 状態（"chat" または "analysis"）を管理
ビュー切り替え時のコンポーネント遷移を制御
components/view-switcher.tsx:
チャットと分析を切り替えるUIインターフェース
タブ形式の切り替えボタンを提供
**確認事項:**
`app-shell.tsx`, `chat-view.tsx`, `analysis-view.tsx` のコードを確認した結果、設計通り各モードがコンポーネントレベルで独立しており、`app-shell.tsx` によって適切に切り替えられていることが確認されました。各ビューは自身の状態管理を持ち、互いに直接的な依存関係はありません。
特記事項
レイジーローディング：
各モードのコンポーネントはlazyとSuspenseを使用して必要なときだけロードされます。
これにより初期ロード時間を短縮し、パフォーマンスを向上させています。
モバイル対応：
use-mobile フックを使用してモバイル表示を検出。
モバイルビューではサイドバーを非表示にするなどの最適化。
状態管理：
ローカルステート（useState）を使用したシンプルな状態管理。
モード切り替え時の状態保持機能。
Next.jsプロジェクト構造の有効活用
ファイルベースのルーティング（App Router）
Server ComponentsとClient Componentsの使い分け
メタデータAPIの活用
"use client" ディレクティブによるクライアントサイドコンポーネントの明示
この構造により、チャットインターフェースから始まり、より詳細な分析へシームレスに移行できる、ユーザーフレンドリーなエクスペリエンスが実現されています。