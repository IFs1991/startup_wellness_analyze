# Gemini APIによるPDFレポート生成機能

## 概要

スタートアップウェルネス分析プラットフォームでは、Google Gemini APIを活用して動的にPDFレポートを生成する機能を実装しています。この機能により、WeasyPrintなどの複雑なシステム依存関係を必要とせず、高品質なレポートを生成できます。

## 主な機能

1. **動的レポート生成**
   - データに基づいた分析レポートを自動生成
   - 企業ウェルネス評価レポート
   - 健康状態と財務パフォーマンスの相関分析

2. **カスタマイズ可能なテンプレート**
   - 複数のレポートテンプレートから選択可能
   - 企業ロゴやブランドカラーの適用
   - セクションの追加・削除のカスタマイズ

3. **マルチメディア対応**
   - Gemini生成のグラフや図表の組み込み
   - データテーブルの視覚的表現
   - エグゼクティブサマリーの自動作成

## 技術詳細

### API構造

```
POST /api/v1/reports/generate
```

### リクエストパラメータ

| パラメータ | 説明 |
|----------|------|
| template_id | 使用するレポートテンプレートのID |
| company_data | 企業データ（JSON形式） |
| period | レポート対象期間（例: "2023Q1", "2023" など） |
| include_sections | 含めるセクションのリスト |
| customization | カスタマイズオプション（ロゴURL、色など） |

### レスポンス

生成されたPDFファイル（application/pdf）が返されます。

## 使用例

### バックエンド（Python）

```python
from api.reports import report_generator

async def generate_company_report(company_id: str, period: str):
    # 企業データの取得
    company_data = await db.get_company_data(company_id, period)

    # レポート生成リクエスト
    report_request = {
        "template_id": "wellness_quarterly",
        "company_data": company_data,
        "period": period,
        "include_sections": [
            "executive_summary",
            "wellness_metrics",
            "financial_correlation",
            "recommendations"
        ],
        "customization": {
            "logo_url": company_data.get("logo_url"),
            "primary_color": "#0066CC",
            "include_benchmarks": True
        }
    }

    # レポート生成
    pdf_data = await report_generator.generate_report(report_request)

    # ファイル保存やクライアントへの送信
    return pdf_data
```

### フロントエンド（React）

```jsx
import React, { useState } from 'react';
import { Button, Select, Checkbox, Spin } from 'antd';
import axios from 'axios';

const ReportGenerator = ({ companyId }) => {
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState('2023Q1');
  const [template, setTemplate] = useState('wellness_quarterly');
  const [sections, setSections] = useState([
    'executive_summary',
    'wellness_metrics',
    'financial_correlation',
    'recommendations'
  ]);

  const generateReport = async () => {
    try {
      setLoading(true);

      const response = await axios.post(
        '/api/v1/reports/generate',
        {
          template_id: template,
          company_id: companyId,
          period: period,
          include_sections: sections,
          customization: {
            include_benchmarks: true
          }
        },
        {
          responseType: 'blob'
        }
      );

      // PDFをダウンロード
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${companyId}_report_${period}.pdf`);
      document.body.appendChild(link);
      link.click();

    } catch (error) {
      console.error('レポート生成エラー:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="report-generator">
      <h2>レポート生成</h2>

      <div className="form-group">
        <label>期間:</label>
        <Select value={period} onChange={setPeriod}>
          <Select.Option value="2023Q1">2023年第1四半期</Select.Option>
          <Select.Option value="2023Q2">2023年第2四半期</Select.Option>
          <Select.Option value="2023Q3">2023年第3四半期</Select.Option>
          <Select.Option value="2023Q4">2023年第4四半期</Select.Option>
        </Select>
      </div>

      {/* 他のフォーム要素 */}

      <Button
        type="primary"
        onClick={generateReport}
        loading={loading}
      >
        レポート生成
      </Button>
    </div>
  );
};
```

## 実装詳細

### Geminiプロンプト構造

レポート生成のためのGeminiプロンプトは、HTMLテンプレートを生成するように設計されています：

```
以下のデータを元に企業ウェルネス評価レポートのHTMLテンプレートを生成してください。

会社データ:
{company_data_json}

期間: {period}

含めるセクション:
{sections_list}

レポート要件:
- プロフェッショナルな企業向けデザイン
- データをグラフィカルに表現
- 簡潔な分析コメント
- アクションにつながる推奨事項
- 企業ロゴと企業カラーの使用

追加要件:
{customization_details}
```

### HTML→PDF変換

生成されたHTMLはPDF変換のためにヘッドレスブラウザ（ChromiumベースのPuppeteer）を使用します。これにより、WeasyPrintなどの複雑なシステム依存関係を避けられます。

## セットアップ

### 必要パッケージ

```
google-generativeai>=0.4.0
pyppeteer  # HTMLからPDF生成
aiofiles   # 非同期ファイル操作
jinja2     # テンプレート操作（必要に応じて）
```

### 環境変数

```
GEMINI_API_KEY=your_gemini_api_key
PUPPETEER_EXECUTABLE_PATH=/path/to/chromium  # 必要に応じて
```

## 注意事項と制限

1. **レポート生成時間**
   - 複雑なレポートの生成には数秒〜数十秒かかる場合があります
   - 非同期処理とバックグラウンドジョブの実装を推奨

2. **スタイリングの制約**
   - Geminiが生成するHTMLは完全に予測可能ではありません
   - テンプレートとCSSでスタイルの一貫性を確保してください

3. **画像とグラフの埋め込み**
   - Base64エンコードされた画像の埋め込みが必要な場合があります
   - グラフは`visualization`APIと組み合わせて使用

## 将来の拡張計画

1. **インタラクティブPDF**
   - ナビゲーションリンクや展開可能なセクションの追加
   - データの掘り下げ機能

2. **複数フォーマット対応**
   - PDF以外にHTML、PowerPoint、Word形式での出力
   - インタラクティブなウェブレポート

3. **AIによるインサイト強化**
   - 重要なパターンの自動検出と強調表示
   - 業界ベンチマークとの比較分析
   - 予測分析の結果を組み込んだレポート