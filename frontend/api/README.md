# フロントエンドAPI統合

このディレクトリには、バックエンドAPIと通信するためのフロントエンドサービスが含まれています。

## 概要

このAPIクライアントは、Startup Wellness分析システムのバックエンドAPIと通信するための統一されたインターフェースを提供します。シングルトンパターンを使用して、アプリケーション全体で一貫したAPIクライアントインスタンスを確保します。

## 主要コンポーネント

- `apiClient.ts` - ベースとなるAPIクライアント。認証トークンの管理やエラーハンドリングを担当
- `types.ts` - APIリクエストとレスポンスの型定義
- `services/` - 各機能領域に特化したサービスクラス

## 利用可能なサービス

以下のサービスが利用可能です：

- `AnalysisService` - データ分析関連のAPI
- `DashboardService` - ダッシュボード管理のAPI
- `ReportService` - レポート生成のAPI
- `StartupDataService` - スタートアップデータ管理のAPI
- `VisualizationService` - データ可視化のAPI（チャート、グラフなど）

## 使用例

### 可視化サービスの使用例

```typescript
import { VisualizationService } from './api/services';
import { ChartConfig, ChartData } from './api/types';

// チャート設定
const config: ChartConfig = {
  chart_type: 'bar',
  title: '四半期売上',
  x_axis_label: '四半期',
  y_axis_label: '売上（百万円）',
  show_legend: true
};

// チャートデータ
const data: ChartData = {
  labels: ['Q1', 'Q2', 'Q3', 'Q4'],
  datasets: [
    {
      label: '2022年',
      data: [12, 19, 15, 22],
      color: '#4285F4'
    },
    {
      label: '2023年',
      data: [15, 22, 18, 25],
      color: '#34A853'
    }
  ]
};

// チャート生成
async function generateChart() {
  try {
    const response = await VisualizationService.generateChart(config, data, 'png');
    if (response.status === 'success' && response.data) {
      console.log('チャートが生成されました:', response.data.url);
      return response.data;
    }
  } catch (error) {
    console.error('チャート生成エラー:', error);
  }
}

// 分析結果の可視化
async function visualizeAnalysis() {
  const analysisResults = {
    // 分析結果データ
  };

  try {
    const response = await VisualizationService.visualizeAnalyzerResults(
      'financial',
      analysisResults,
      'bar',
      { title: '財務分析結果' }
    );

    if (response.status === 'success' && response.data) {
      console.log('分析結果が可視化されました:', response.data.url);
      return response.data;
    }
  } catch (error) {
    console.error('分析結果の可視化エラー:', error);
  }
}
```

### バックグラウンド処理の使用例

```typescript
import { VisualizationService } from './api/services';

async function generateLargeChartInBackground() {
  // 大きなチャートの設定とデータ
  const config = { /* ... */ };
  const data = { /* ... */ };

  try {
    // バックグラウンドでチャート生成を開始
    const jobResponse = await VisualizationService.generateChartBackground(config, data);

    if (jobResponse.status === 'success' && jobResponse.data) {
      const jobId = jobResponse.data.job_id;
      console.log('バックグラウンドジョブが開始されました:', jobId);

      // 定期的にステータスを確認
      const checkStatus = async () => {
        const statusResponse = await VisualizationService.getChartStatus(jobId);

        if (statusResponse.data?.status === 'completed') {
          console.log('チャート生成が完了しました:', statusResponse.data.result);
          return statusResponse.data.result;
        } else if (statusResponse.data?.status === 'failed') {
          console.error('チャート生成に失敗しました:', statusResponse.data.error);
          throw new Error(statusResponse.data.error);
        } else {
          // まだ処理中の場合は少し待ってから再確認
          console.log('処理中...');
          setTimeout(checkStatus, 2000);
        }
      };

      // 最初のステータスチェックを開始
      setTimeout(checkStatus, 2000);
    }
  } catch (error) {
    console.error('バックグラウンドチャート生成エラー:', error);
  }
}
```

## 注意事項

- 認証が必要なAPIエンドポイントにアクセスする前に、ユーザーがログインしていることを確認してください。
- 大きなデータセットを処理する場合は、バックグラウンド処理APIを使用することを検討してください。
- APIレスポンスのエラーハンドリングを適切に行ってください。