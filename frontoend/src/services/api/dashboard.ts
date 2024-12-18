import { AxiosInstance } from 'axios';

// ダッシュボードで使用するデータの型定義
interface DashboardMetrics {
  totalUsers: number;
  activeUsers: number;
  revenue: number;
  growth: number;
}

interface GraphData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
  }[];
}

// ダッシュボードサービスクラス
export class DashboardService {
  private api: AxiosInstance;

  constructor(api: AxiosInstance) {
    this.api = api;
  }

  // ダッシュボードの主要メトリクスを取得
  async getMetrics(): Promise<DashboardMetrics> {
    const response = await this.api.get<DashboardMetrics>('/api/dashboard/metrics');
    return response.data;
  }

  // グラフデータを取得
  async getGraphData(period: string = '7d'): Promise<GraphData> {
    const response = await this.api.get<GraphData>(`/api/dashboard/graph?period=${period}`);
    return response.data;
  }

  // ユーザーアクティビティを取得
  async getUserActivity(limit: number = 10): Promise<any[]> {
    const response = await this.api.get(`/api/dashboard/activity?limit=${limit}`);
    return response.data;
  }
}

// エクスポートするインスタンスを作成
import axios from 'axios';

// APIクライアントの作成
const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// リクエストインターセプターの追加
apiClient.interceptors.request.use((config) => {
  // JWTトークンがあれば追加
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

const dashboardService = new DashboardService(apiClient);
export default dashboardService;