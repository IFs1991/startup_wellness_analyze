import { useState, useEffect, useCallback } from 'react';
import { useWebSocketConnection } from './useWebSocketConnection';
import { useToast } from './use-toast';

interface WellnessScoreData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    borderColor: string;
    backgroundColor: string;
    tension: number;
    borderDash?: number[];
  }[];
}

interface CompanyMetrics {
  averageScore: number;
  scoreChange: number;
  topPerformers: number;
  engagementRate: number;
}

interface CompanySummary {
  id: string;
  name: string;
  score: number;
  change: number;
}

interface AnalysisInsight {
  title: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  metrics: string[];
}

export interface DashboardData {
  wellnessScores?: WellnessScoreData;
  metrics?: CompanyMetrics;
  recentActivities?: Array<{
    id: string;
    type: string;
    companyName: string;
    description: string;
    timestamp: string;
  }>;
  topCompanies?: CompanySummary[];
  analysisInsights?: AnalysisInsight[];
  loading: boolean;
  error: Error | null;
}

/**
 * ダッシュボードデータを取得するカスタムフック
 * WebSocketを使用してリアルタイムデータを取得します
 *
 * @returns ダッシュボードデータと状態、更新関数
 */
export const useDashboardData = (): DashboardData & {
  refreshData: () => void;
  getDataForPeriod: (period: string) => void;
} => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const [wellnessScores, setWellnessScores] = useState<WellnessScoreData | undefined>(undefined);
  const [metrics, setMetrics] = useState<CompanyMetrics | undefined>(undefined);
  const [recentActivities, setRecentActivities] = useState<DashboardData['recentActivities']>(undefined);
  const [topCompanies, setTopCompanies] = useState<CompanySummary[] | undefined>(undefined);
  const [analysisInsights, setAnalysisInsights] = useState<AnalysisInsight[] | undefined>(undefined);

  // WebSocket接続を確立
  const {
    status,
    sendMessage,
    messages,
    error: wsError
  } = useWebSocketConnection('dashboard');

  // エラー時の処理
  useEffect(() => {
    if (wsError) {
      setError(wsError);
      setLoading(false);

      toast({
        title: 'データ取得エラー',
        description: 'ダッシュボードデータの取得に失敗しました',
        variant: 'destructive'
      });
    }
  }, [wsError, toast]);

  // 接続状態の監視
  useEffect(() => {
    if (status === 'connected') {
      // 接続成功時にデータをリクエスト
      sendMessage('get_dashboard_data');
    } else if (status === 'disconnected' || status === 'error') {
      setLoading(false);
    }
  }, [status, sendMessage]);

  // メッセージの処理
  useEffect(() => {
    if (messages && messages.length > 0) {
      const latestMessage = messages[messages.length - 1];

      if (latestMessage.type === 'dashboard_data') {
        const data = latestMessage.data;

        // 各データの設定
        if (data.wellnessScores) {
          setWellnessScores(data.wellnessScores);
        }

        if (data.metrics) {
          setMetrics(data.metrics);
        }

        if (data.recentActivities) {
          setRecentActivities(data.recentActivities);
        }

        if (data.topCompanies) {
          setTopCompanies(data.topCompanies);
        }

        if (data.analysisInsights) {
          setAnalysisInsights(data.analysisInsights);
        }

        setLoading(false);
      }
    }
  }, [messages]);

  // データの更新をリクエストする関数
  const refreshData = useCallback(() => {
    if (status === 'connected') {
      setLoading(true);
      sendMessage('get_dashboard_data');
    } else {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
    }
  }, [status, sendMessage, toast]);

  // 特定の期間のデータをリクエスト
  const getDataForPeriod = useCallback((period: string) => {
    if (status === 'connected') {
      setLoading(true);
      sendMessage('get_dashboard_data', { period });
    }
  }, [status, sendMessage]);

  return {
    wellnessScores,
    metrics,
    recentActivities,
    topCompanies,
    analysisInsights,
    loading,
    error,
    refreshData,
    getDataForPeriod
  } as DashboardData & {
    refreshData: () => void;
    getDataForPeriod: (period: string) => void;
  };
};