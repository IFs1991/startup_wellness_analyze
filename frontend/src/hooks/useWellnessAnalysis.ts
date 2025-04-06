import { useState, useEffect, useCallback } from 'react';
import { useWebSocketConnection } from './useWebSocketConnection';
import { useToast } from './use-toast';

export interface CorrelationData {
  matrix: number[][];
  variables: string[];
  insights: string[];
}

export interface ClusterAnalysisData {
  points: Array<{
    x: number;
    y: number;
    clusterId: string;
  }>;
  clusters: Array<{
    id: string;
    name: string;
    count: number;
    percentage: number;
    color: string;
    features: Record<string, any>;
    insights: string[];
  }>;
  xAxisLabel: string;
  yAxisLabel: string;
}

export interface TimeSeriesData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    borderColor: string;
    backgroundColor: string;
    tension: number;
    borderDash?: number[];
  }>;
  insights: string[];
}

export interface SurvivalAnalysisData {
  survivalCurve: Array<{
    time: number;
    survival: number;
  }>;
  riskFactors: Array<{
    factor: string;
    importance: number;
    description: string;
  }>;
  segments: Array<{
    name: string;
    color: string;
    data: Array<{
      time: number;
      survival: number;
    }>;
  }>;
  medianSurvivalTime: number;
  insights: string[];
}

export interface PcaData {
  points: Array<{
    x: number;
    y: number;
    id: string;
    name: string;
  }>;
  loadings: Array<{
    feature: string;
    pc1: number;
    pc2: number;
  }>;
  explainedVariance: {
    pc1: number;
    pc2: number;
  };
  insights: string[];
}

export interface DescriptiveStats {
  mean: Record<string, number>;
  median: Record<string, number>;
  std: Record<string, number>;
  min: Record<string, number>;
  max: Record<string, number>;
  insights: string[];
}

export interface AnalysisFilter {
  timeRange?: string;
  departments?: string[];
  jobRoles?: string[];
  ageGroups?: string[];
  genders?: string[];
}

export interface WellnessAnalysisData {
  correlationData?: CorrelationData;
  clusterData?: ClusterAnalysisData;
  timeSeriesData?: TimeSeriesData;
  survivalData?: SurvivalAnalysisData;
  pcaData?: PcaData;
  descriptiveStats?: DescriptiveStats;
  loading: boolean;
  error: Error | null;
}

/**
 * ウェルネス分析データを取得するカスタムフック
 *
 * @param companyId 企業ID（指定しない場合は全体データ）
 * @param filters 分析フィルター
 * @returns 分析データと状態
 */
export const useWellnessAnalysis = (
  companyId?: string,
  filters: AnalysisFilter = {}
): WellnessAnalysisData & {
  refreshData: () => void;
  applyFilters: (filters: AnalysisFilter) => void;
} => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [currentFilters, setCurrentFilters] = useState<AnalysisFilter>(filters);

  const [correlationData, setCorrelationData] = useState<CorrelationData | undefined>(undefined);
  const [clusterData, setClusterData] = useState<ClusterAnalysisData | undefined>(undefined);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData | undefined>(undefined);
  const [survivalData, setSurvivalData] = useState<SurvivalAnalysisData | undefined>(undefined);
  const [pcaData, setPcaData] = useState<PcaData | undefined>(undefined);
  const [descriptiveStats, setDescriptiveStats] = useState<DescriptiveStats | undefined>(undefined);

  // エンドポイント作成
  const endpoint = companyId ? `analysis/company/${companyId}` : 'analysis';

  // WebSocket接続を確立
  const {
    status,
    sendMessage,
    messages,
    error: wsError
  } = useWebSocketConnection(endpoint);

  // エラー時の処理
  useEffect(() => {
    if (wsError) {
      setError(wsError);
      setLoading(false);

      toast({
        title: 'データ取得エラー',
        description: '分析データの取得に失敗しました',
        variant: 'destructive'
      });
    }
  }, [wsError, toast]);

  // 接続状態の監視
  useEffect(() => {
    if (status === 'connected') {
      // 接続成功時にデータをリクエスト
      requestAnalysisData();
    } else if (status === 'disconnected' || status === 'error') {
      setLoading(false);
    }
  }, [status]);

  // 分析データをリクエストする関数
  const requestAnalysisData = useCallback(() => {
    if (status === 'connected') {
      setLoading(true);

      const requestPayload = {
        company_id: companyId,
        filters: currentFilters
      };

      sendMessage('get_analysis_data', requestPayload);
    }
  }, [status, sendMessage, companyId, currentFilters]);

  // フィルターを適用する関数
  const applyFilters = useCallback((newFilters: AnalysisFilter) => {
    setCurrentFilters(prev => ({
      ...prev,
      ...newFilters
    }));

    if (status === 'connected') {
      setLoading(true);

      const requestPayload = {
        company_id: companyId,
        filters: {
          ...currentFilters,
          ...newFilters
        }
      };

      sendMessage('get_analysis_data', requestPayload);
    } else {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
    }
  }, [status, sendMessage, companyId, currentFilters, toast]);

  // メッセージの処理
  useEffect(() => {
    if (messages && messages.length > 0) {
      const latestMessage = messages[messages.length - 1];

      if (latestMessage.type === 'analysis_data') {
        const data = latestMessage.data;

        // 各データの設定
        if (data.correlationData) {
          setCorrelationData(data.correlationData);
        }

        if (data.clusterData) {
          setClusterData(data.clusterData);
        }

        if (data.timeSeriesData) {
          setTimeSeriesData(data.timeSeriesData);
        }

        if (data.survivalData) {
          setSurvivalData(data.survivalData);
        }

        if (data.pcaData) {
          setPcaData(data.pcaData);
        }

        if (data.descriptiveStats) {
          setDescriptiveStats(data.descriptiveStats);
        }

        setLoading(false);
      }
    }
  }, [messages]);

  // データの更新をリクエストする関数
  const refreshData = useCallback(() => {
    if (status === 'connected') {
      requestAnalysisData();
    } else {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
    }
  }, [status, requestAnalysisData, toast]);

  return {
    correlationData,
    clusterData,
    timeSeriesData,
    survivalData,
    pcaData,
    descriptiveStats,
    loading,
    error,
    refreshData,
    applyFilters
  };
};