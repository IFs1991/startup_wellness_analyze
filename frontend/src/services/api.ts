import axios, { AxiosError, AxiosInstance, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import { getAuth } from 'firebase/auth';

// APIクライアントの基本設定
const apiClient: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30秒タイムアウト設定
});

// リクエストインターセプター - 認証トークンの付与
apiClient.interceptors.request.use(
  async (config: InternalAxiosRequestConfig) => {
    try {
      const auth = getAuth();
      const user = auth.currentUser;

      if (user) {
        const token = await user.getIdToken(true); // forceRefreshをtrueに設定して最新のトークンを取得
        if (config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }
      }
      return config;
    } catch (error) {
      console.error('認証トークン取得エラー:', error);
      return config;
    }
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

// レスポンスインターセプター - エラーハンドリング
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  async (error: AxiosError) => {
    const status = error.response?.status;

    // 認証エラーのハンドリング
    if (status === 401) {
      console.error('認証エラー: ログインが必要です');
      // ユーザーの状態を確認
      const auth = getAuth();
      if (auth.currentUser) {
        // ユーザーはログインしているがトークンが無効
        try {
          // トークンを更新して再試行
          await auth.currentUser.getIdToken(true);
          // 元のリクエストを再試行
          if (error.config) {
            return apiClient(error.config);
          }
        } catch (refreshError) {
          console.error('トークン更新エラー:', refreshError);
          // トークン更新に失敗した場合はログアウト処理
          await auth.signOut();
          window.location.href = '/login';
        }
      } else {
        // ユーザーがログインしていない場合
        window.location.href = '/login';
      }
    }

    // 権限エラーのハンドリング
    if (status === 403) {
      console.error('権限エラー: アクセス権がありません');
    }

    // サーバーエラーのハンドリング
    if (status && status >= 500) {
      console.error('サーバーエラーが発生しました');
    }

    return Promise.reject(error);
  }
);

// サブスクリプション関連のAPI
export const subscriptionApi = {
  // プラン一覧を取得
  getPlans: () => apiClient.get('/subscriptions/plans'),

  // 無料トライアルに登録
  registerFreeTrial: (userData: any) => apiClient.post('/subscriptions/register/free-trial', userData),

  // チェックアウトセッションを作成
  createCheckout: (planId: string) => apiClient.post('/subscriptions/checkout', { plan_id: planId }),

  // サブスクリプションの状態を取得
  getStatus: () => apiClient.get('/subscriptions/status'),

  // 支払い方法を取得
  getPaymentMethod: () => apiClient.get('/subscriptions/payment-method'),

  // 請求書の履歴を取得
  getInvoices: () => apiClient.get('/subscriptions/invoices'),

  // サブスクリプションをキャンセル
  cancelSubscription: (subscriptionId: string, atPeriodEnd: boolean = true) =>
    apiClient.post('/subscriptions/cancel', {
      subscription_id: subscriptionId,
      at_period_end: atPeriodEnd
    }),

  // サブスクリプションを再アクティベート
  reactivateSubscription: (subscriptionId: string) =>
    apiClient.post('/subscriptions/reactivate', { subscription_id: subscriptionId }),
};

// 分析関連のAPI
export const analysisApi = {
  // ウェルネススコアを計算
  calculateWellnessScore: (data: any) => apiClient.post('/analysis/wellness-score', data),

  // 相関分析を実行
  analyzeCorrelation: (data: any) => apiClient.post('/analysis/correlation', data),

  // クラスター分析を実行
  analyzeClusters: (data: any) => apiClient.post('/analysis/cluster', data),

  // 時系列分析を実行
  analyzeTimeSeries: (data: any) => apiClient.post('/analysis/time-series', data),

  // 生存時間分析を実行
  analyzeSurvival: (data: any) => apiClient.post('/analysis/survival', data),

  // スタートアップ生存分析を実行
  analyzeStartupSurvival: (data: any) => apiClient.post('/analysis/startup-survival', data),

  // 主成分分析を実行
  analyzePCA: (data: any) => apiClient.post('/analysis/pca', data),
};

// レポート生成API
export const reportApi = {
  // レポートを生成
  generateReport: (data: any) => apiClient.post('/report/generate', data),

  // AI要約を生成
  generateAISummary: (data: any) => apiClient.post('/ai/summarize', data),
};

// 価格プラン関連のAPI
export const pricingApi = {
  // パーソナライズされた価格情報を取得
  getPersonalizedPricing: () => apiClient.get('/pricing/personalized'),

  // トライアル状態を取得
  getTrialStatus: () => apiClient.get('/pricing/trial-status'),
};

// レポート生成APIリクエスト関数
export const generateReport = async (data: {
  template_id: string;
  company_data: any;
  period: string;
  include_sections: string[];
  customization?: any;
  format: string;
}) => {
  try {
    const response = await fetch('/api/v1/reports/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'レポート生成に失敗しました');
    }

    return await response.json();
  } catch (error) {
    console.error('Report generation error:', error);
    throw error;
  }
};

// レポートのステータス確認API
export const checkReportStatus = async (reportId: string) => {
  try {
    const response = await fetch(`/api/v1/reports/status/${reportId}`);

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'レポートステータスの取得に失敗しました');
    }

    return await response.json();
  } catch (error) {
    console.error('Report status check error:', error);
    throw error;
  }
};

// デフォルトのAPIクライアントをエクスポート
export default apiClient;