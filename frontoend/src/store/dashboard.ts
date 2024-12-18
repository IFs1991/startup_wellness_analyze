import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// ダッシュボードの指標データの型定義
interface Metrics {
  totalUsers: number;
  activeUsers: number;
  averageEngagement: number;
  completionRate: number;
}

// グラフデータの型定義
interface ChartData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
  }>;
}

// ダッシュボードの期間設定の型定義
type TimeRange = '7d' | '30d' | '90d' | '1y';

// ダッシュボードの状態を定義するインターフェース
interface DashboardState {
  // 基本的な状態
  isLoading: boolean;
  error: string | null;
  timeRange: TimeRange;
  metrics: Metrics | null;
  chartData: ChartData | null;

  // アクション
  setTimeRange: (range: TimeRange) => void;
  setMetrics: (metrics: Metrics) => void;
  setChartData: (data: ChartData) => void;
  setError: (error: string | null) => void;
  setLoading: (loading: boolean) => void;
  resetState: () => void;

  // データ取得アクション
  fetchDashboardData: () => Promise<void>;
}

// 初期状態の定義
const initialState = {
  isLoading: false,
  error: null,
  timeRange: '7d' as TimeRange,
  metrics: null,
  chartData: null,
};

// ダッシュボードストアの作成
export const useDashboardStore = create<DashboardState>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        // 基本的なステート更新アクション
        setTimeRange: (range) => set({ timeRange: range }),
        setMetrics: (metrics) => set({ metrics }),
        setChartData: (data) => set({ chartData: data }),
        setError: (error) => set({ error }),
        setLoading: (loading) => set({ isLoading: loading }),
        resetState: () => set(initialState),

        // データ取得アクション
        fetchDashboardData: async () => {
          const { timeRange } = get();
          set({ isLoading: true, error: null });

          try {
            // メトリクスデータの取得
            const metricsResponse = await fetch(`/api/dashboard/metrics?range=${timeRange}`);
            if (!metricsResponse.ok) {
              throw new Error('メトリクスデータの取得に失敗しました');
            }
            const metricsData: Metrics = await metricsResponse.json();

            // チャートデータの取得
            const chartResponse = await fetch(`/api/dashboard/chart?range=${timeRange}`);
            if (!chartResponse.ok) {
              throw new Error('チャートデータの取得に失敗しました');
            }
            const chartData: ChartData = await chartResponse.json();

            // 状態を更新
            set({
              metrics: metricsData,
              chartData: chartData,
              isLoading: false,
            });
          } catch (error) {
            set({
              error: error instanceof Error ? error.message : '予期せぬエラーが発生しました',
              isLoading: false,
            });
          }
        },
      }),
      {
        name: 'dashboard-storage', // ローカルストレージのキー
        partialize: (state) => ({
          timeRange: state.timeRange, // 永続化する状態を選択
        }),
      }
    )
  )
);

// 選択可能な期間オプション
export const timeRangeOptions = [
  { value: '7d', label: '過去7日間' },
  { value: '30d', label: '過去30日間' },
  { value: '90d', label: '過去90日間' },
  { value: '1y', label: '過去1年' },
] as const;

// ダッシュボードで使用する定数
export const DASHBOARD_CONSTANTS = {
  REFRESH_INTERVAL: 5 * 60 * 1000, // 5分ごとに更新
  DEFAULT_TIME_RANGE: '7d' as TimeRange,
  CHART_COLORS: {
    primary: '#4F46E5',
    secondary: '#10B981',
    accent: '#F59E0B',
  },
} as const;