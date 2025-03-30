# Gemini APIを活用した動的可視化ガイド

## 概要

スタートアップウェルネス分析プラットフォームでは、Google Gemini APIを活用して動的なデータ可視化を実現しています。これにより、システムライブラリの依存関係を減らしつつ、高品質なグラフや視覚化を提供しています。

## メリット

1. **システム依存関係の軽減**
   - matplotlibやOpenCVなどの重いグラフィックライブラリが不要
   - インストールと設定が簡素化
   - コンテナサイズの削減

2. **柔軟な可視化表現**
   - 自然言語による指示でカスタマイズ可能
   - 多様なグラフタイプとスタイルに対応
   - デザインの一貫性を確保

3. **開発効率の向上**
   - グラフライブラリAPIの複雑な操作が不要
   - 視覚化コードのメンテナンスコスト削減
   - 短時間で高品質な可視化を作成

## 対応する可視化タイプ

- 棒グラフ（縦棒・横棒）
- 折れ線グラフ
- 散布図
- 円グラフ
- ヒートマップ
- 複合グラフ
- ダッシュボード（開発中）

## 使用シナリオ

### 1. データ分析結果の視覚化

財務データや従業員健康データの分析結果を視覚的に表現する際に使用します。例えば、健康スコアと生産性の相関を示す散布図や、時間経過による健康指標の変化を示す折れ線グラフなどです。

### 2. レポート生成

定期的なウェルネスレポートの中で、主要指標や傾向を視覚的に表現するためのグラフを自動生成します。フォーマットを統一しつつ、データに応じた最適なビジュアルを提供します。

### 3. リアルタイムダッシュボード

企業の健康状態をリアルタイムで監視するダッシュボードにおいて、現在の状況を示すグラフや図を動的に生成します。

## 実装詳細

### APIエンドポイント

現在、以下のAPIエンドポイントが実装されています：

- `POST /api/v1/visualization/chart` - 単一のチャートを生成
- `POST /api/v1/visualization/dashboard` - 複数チャートのダッシュボードを生成（開発中）

### 利用方法

#### バックエンド（Python）

```python
from api.visualization import visualizer

# チャート生成
chart_data = {
    "data": [
        {"category": "健康スコア", "value": 8.5},
        {"category": "生産性", "value": 7.2},
        {"category": "ワークライフバランス", "value": 6.8},
        {"category": "ストレスレベル", "value": 4.2}
    ],
    "chart_type": "bar",
    "title": "ウェルネス指標",
    "description": "主要なウェルネス指標の比較"
}

# 画像データを取得
image_data = await visualizer.generate_chart(chart_data)

# 画像データの保存や返却
with open("wellness_chart.png", "wb") as f:
    f.write(image_data)
```

#### フロントエンド（React）

```jsx
import React from 'react';
import GeminiChart from 'components/GeminiChart';

const WellnessReport = () => {
  const wellnessData = [
    {quarter: "Q1", score: 7.2},
    {quarter: "Q2", score: 7.5},
    {quarter: "Q3", score: 8.1},
    {quarter: "Q4", score: 8.4}
  ];

  return (
    <div className="report-container">
      <h1>四半期ウェルネスレポート</h1>

      <div className="chart-section">
        <GeminiChart
          data={wellnessData}
          chartType="line"
          title="ウェルネススコアの推移"
          description="四半期ごとのウェルネススコア変化"
          theme="professional"
        />
      </div>

      {/* レポートの他のセクション */}
    </div>
  );
};
```

### リクエストパラメータ

| パラメータ名 | 型 | 説明 | 必須 |
|--------------|----|--------------------|--------|
| data | Object/Array | グラフ化するデータ | はい |
| chart_type | String | チャートタイプ（bar, line, scatter, pie, heatmap など） | はい |
| title | String | チャートのタイトル | はい |
| description | String | チャートの説明（Geminiプロンプトに含まれる） | いいえ |
| width | Number | 画像の幅（ピクセル） | いいえ |
| height | Number | 画像の高さ（ピクセル） | いいえ |
| language | String | ラベルの言語（ja/en） | いいえ |
| theme | String | テーマ（professional, minimal, dark, light） | いいえ |

## セットアップガイド

### 1. 必要なパッケージ

以下のPythonパッケージが必要です：

```
google-generativeai>=0.4.0
pillow
httpx
```

`backend/environment.yml`に既に含まれています。

### 2. 環境変数の設定

`.env`ファイルまたは環境変数として、以下を設定してください：

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. APIキーの取得

1. [Google AI Studio](https://makersuite.google.com/)にアクセス
2. アカウントを作成または既存のGoogleアカウントでログイン
3. APIキーを生成
4. 生成されたキーを環境変数にセット

## 注意事項

1. **APIキーの扱い**
   - APIキーは機密情報として安全に管理してください
   - 本番環境ではシークレット管理サービスの使用を推奨

2. **API利用制限**
   - Gemini APIには利用制限（レート制限）があります
   - 多数のグラフを短時間で生成する場合はキャッシュの実装を検討

3. **インターネット接続**
   - この機能はインターネット接続が必要です
   - オフライン環境では使用できません

4. **グラフのカスタマイズ**
   - さらに詳細なカスタマイズが必要な場合は、プロンプトの構造を変更してください
   - `visualization.py`の`generate_chart`メソッドにあるプロンプトテンプレートを調整可能

## 例：プロンプト構造

Gemini APIに送信されるプロンプトは以下のような構造になっています：

```
以下のデータを使用して{chart_type}チャートを生成してください。

タイトル: {title}

データ:
{data_json}

要件:
- {language}でラベルを表示
- 幅: {width}px、高さ: {height}px
- {color_theme}
- データを明確に表示
- 軸ラベルとタイトルを適切に配置
- 凡例を見やすく配置

追加要件: {description}
```

## 今後の拡張計画

1. **ダッシュボード生成機能の完成**
   - 複数の関連チャートを1つの画像として配置
   - レイアウトのカスタマイズオプション

2. **インタラクティブな可視化**
   - SVGフォーマットでの出力
   - クライアント側での簡単なインタラクション

3. **テンプレートギャラリー**
   - よく使われるグラフテンプレートの事前定義
   - ワンクリックでの適用