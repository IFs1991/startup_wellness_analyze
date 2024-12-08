import { getAuth } from 'firebase/auth';
import {
  useQuery,
  useMutation,
  UseQueryResult,
  UseMutationResult,
  QueryClient
} from '@tanstack/react-query';

// 型定義 - ダッシュボード設定
export interface DashboardConfig {
  title: string;
  description?: string;
  widgets: Array<Record<string, any>>;
  layout: Record<string, any>;
}

// 型定義 - グラフ設定
export interface GraphConfig {
  type: string;
  title: string;
  data_source: string;
  settings: Record<string, any>;
  filters?: Array<{
    field: string;
    operator: string;
    value: any;
  }>;
}

// 型定義 - 可視化レスポンス
export interface VisualizationResponse {
  id: string;
  created_at: string;
  updated_at: string;
  config: Record<string, any>;
  data: Record<string, any>;
  created_by: string;
}

// APIレスポンスの型定義
export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

// 環境変数からAPIのベースURLを取得
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// 認証トークンを取得する関数
async function getAuthToken(): Promise<string> {
  const auth = getAuth();
  const user = auth.currentUser;

  if (!user) {
    throw new Error('ユーザーがログインしていません');
  }

  return user.getIdToken();
}

// 認証付きのAPIリクエストを行うヘルパー関数
async function fetchWithAuth(
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> {
  try {
    const token = await getAuthToken();
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
      ...options.headers,
    };

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'APIリクエストが失敗しました');
    }

    return response;
  } catch (error) {
    console.error('API通信エラー:', error);
    throw error;
  }
}

// APIクライアント
export const api = {
  // ダッシュボード関連の操作
  async createDashboard(config: DashboardConfig): Promise<ApiResponse<VisualizationResponse>> {
    const response = await fetchWithAuth('/dashboard/create', {
      method: 'POST',
      body: JSON.stringify(config),
    });
    return response.json();
  },

  // グラフ関連の操作
  async createGraph(config: GraphConfig): Promise<ApiResponse<VisualizationResponse>> {
    const response = await fetchWithAuth('/graph/create', {
      method: 'POST',
      body: JSON.stringify(config),
    });
    return response.json();
  },

  // ユーザーの可視化データを取得
  async getUserVisualizations(): Promise<ApiResponse<VisualizationResponse[]>> {
    const response = await fetchWithAuth('/visualizations/user');
    return response.json();
  },

  // 分析関連の操作
  analysis: {
    async analyze(data: any): Promise<ApiResponse<any>> {
      const response = await fetchWithAuth('/analysis/process', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      return response.json();
    }
  },

  // データ入力関連の操作
  dataInput: {
    async saveData(data: any): Promise<ApiResponse<any>> {
      const response = await fetchWithAuth('/data_input/save', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      return response.json();
    }
  },

  // データ処理関連の操作
  dataProcessing: {
    async processData(data: any): Promise<ApiResponse<any>> {
      const response = await fetchWithAuth('/data_processing/process', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      return response.json();
    }
  },

  // 予測関連の操作
  prediction: {
    async getPrediction(params: any): Promise<ApiResponse<any>> {
      const response = await fetchWithAuth('/prediction/get', {
        method: 'POST',
        body: JSON.stringify(params),
      });
      return response.json();
    }
  },

  // レポート生成関連の操作
  reportGeneration: {
    async generateReport(config: any): Promise<ApiResponse<any>> {
      const response = await fetchWithAuth('/report_generation/create', {
        method: 'POST',
        body: JSON.stringify(config),
      });
      return response.json();
    }
  },

  // ヘルスチェック
  async healthCheck(): Promise<{ status: string; version: string; timestamp: string }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },
};

// React Queryを使用したカスタムフック
export const useUserVisualizations = (): UseQueryResult<ApiResponse<VisualizationResponse[]>, Error> => {
  return useQuery({
    queryKey: ['visualizations'],
    queryFn: () => api.getUserVisualizations(),
  });
};

export const useCreateDashboard = (): UseMutationResult<
  ApiResponse<VisualizationResponse>,
  Error,
  DashboardConfig,
  unknown
> => {
  return useMutation({
    mutationFn: (config: DashboardConfig) => api.createDashboard(config),
  });
};

export const useCreateGraph = (): UseMutationResult<
  ApiResponse<VisualizationResponse>,
  Error,
  GraphConfig,
  unknown
> => {
  return useMutation({
    mutationFn: (config: GraphConfig) => api.createGraph(config),
  });
};

// エラーハンドリングのためのユーティリティ関数
export const handleApiError = (error: any): string => {
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
};