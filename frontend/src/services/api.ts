import axios, { AxiosError, AxiosInstance, AxiosRequestConfig, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import { getAuth } from 'firebase/auth';

// APIクライアントの基本設定
const apiClient: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// リクエストインターセプター - 認証トークンの付与
apiClient.interceptors.request.use(
  async (config: InternalAxiosRequestConfig) => {
    try {
      const auth = getAuth();
      const user = auth.currentUser;

      if (user) {
        const token = await user.getIdToken();
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
      // ログイン画面へのリダイレクトなどの処理
      window.location.href = '/login';
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

// 価格プラン関連のAPI
export const pricingApi = {
  // パーソナライズされた価格情報を取得
  getPersonalizedPricing: () => apiClient.get('/pricing/personalized'),

  // トライアル状態を取得
  getTrialStatus: () => apiClient.get('/pricing/trial-status'),
};

// デフォルトのAPIクライアントをエクスポート
export default apiClient;