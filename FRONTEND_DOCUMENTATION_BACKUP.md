# フロントエンド開発ドキュメント

## 技術スタック
### フレームワーク/ライブラリ
- React 18.3.1
- TypeScript
- Vite（ビルドツール）
- TailwindCSS（スタイリング）

### 主要な依存関係
- `@reduxjs/toolkit` & `react-redux`：状態管理
- `react-router-dom`：ルーティング
- `react-query`：APIデータフェッチング
- `recharts`：データ可視化
- `zod`：バリデーション
- `@radix-ui`：UIコンポーネント群

## プロジェクト構造
```
frontend/
├── src/
│   ├── components/     # 再利用可能なUIコンポーネント
│   ├── pages/         # ページコンポーネント
│   ├── store/         # Reduxストア関連
│   ├── lib/           # ユーティリティ関数
│   ├── hooks/         # カスタムフック
│   └── App.tsx        # メインアプリケーション
```

## 開発環境セットアップ
1. 開発サーバー起動: `npm run dev`
2. ビルド: `npm run build`
3. リント: `npm run lint`

## 重要な設定ファイル
- `vite.config.ts`：Viteの設定
- `tailwind.config.js`：TailwindCSSの設定
- `tsconfig.json`：TypeScriptの設定
- `.env.local`：環境変数

## デザインシステム
- Radix UIコンポーネントをベースとしたUIライブラリを使用
- TailwindCSSによるスタイリング
- アニメーション用の`tailwindcss-animate`を使用

## 状態管理
- Redux Toolkitを使用した中央集中型の状態管理
- React Queryによるサーバーステート管理

## 注意点
1. TypeScriptの厳格な型チェックが有効
2. ESLintによるコード品質管理
3. コンポーネントはアトミックデザインの原則に従っている可能性が高い
4. 環境変数は`.env.local`で管理