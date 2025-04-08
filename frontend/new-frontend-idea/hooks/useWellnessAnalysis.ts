import { useState, useEffect, useCallback } from 'react';
import { useWebSocketConnection } from './useWebSocketConnection';
import { useToast } from './use-toast';

/**
 * 分析フィルターのインターフェース
 */
interface AnalysisFilter {
  startDate?: string;
  endDate?: string;
  department?: string;
  ageRange?: string;
  gender?: string;
}

/**
 * 相関データのインターフェース
 */
interface CorrelationData {
  variables: string[];
  matrix: number[][];
  pValues: number[][];
}

/**
 * クラスター分析データのインターフェース
 */
interface ClusterAnalysisData {
  clusters: number[];
  centroids: number[][];
  silhouetteScore: number;
}

/**
 * 時系列データのインターフェース
 */
interface TimeSeriesData {
  dates: string[];
  values: number[];
  trend: number[];
  seasonality: number[];
}

/**
 * 生存分析データのインターフェース
 */
interface SurvivalAnalysisData {
  timePoints: number[];
  survivalProbabilities: number[];
  hazardRates: number[];
}

/**
 * PCAデータのインターフェース
 */
interface PcaData {
  components: number[][];
  explainedVariance: number[];
  loadings: number[][];
}

/**
 * 記述統計データのインターフェース
 */
interface DescriptiveStats {
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  quartiles: number[];
}

/**
 * ウェルネス分析データのインターフェース
 */
interface WellnessAnalysisData {
  correlations: CorrelationData;
  clusters: ClusterAnalysisData;
  timeSeries: TimeSeriesData;
  survival: SurvivalAnalysisData;
  pca: PcaData;
  stats: Record<string, DescriptiveStats>;
  isLoading: boolean;
  error: string | null;
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
  const [data, setData] = useState<WellnessAnalysisData>({
    correlations: {
      variables: [],
      matrix: [],
      pValues: []
    },
    clusters: {
      clusters: [],
      centroids: [],
      silhouetteScore: 0
    },
    timeSeries: {
      dates: [],
      values: [],
      trend: [],
      seasonality: []
    },
    survival: {
      timePoints: [],
      survivalProbabilities: [],
      hazardRates: []
    },
    pca: {
      components: [],
      explainedVariance: [],
      loadings: []
    },
    stats: {},
    isLoading: true,
    error: null
  });

  const { sendMessage } = useWebSocketConnection('wellness-analysis', {
    autoReconnect: true,
    reconnectInterval: 5000,
    maxReconnectAttempts: 5
  });
  const { toast } = useToast();

  const refreshData = useCallback(() => {
    if (!companyId) return;
    sendMessage(JSON.stringify({
      type: 'get_wellness_analysis',
      payload: { companyId, filters }
    }));
  }, [companyId, filters, sendMessage]);

  const applyFilters = useCallback((newFilters: AnalysisFilter) => {
    setData(prev => ({ ...prev, isLoading: true, error: null }));
    if (companyId) {
      sendMessage(JSON.stringify({
        type: 'get_wellness_analysis',
        payload: { companyId, filters: newFilters }
      }));
    }
  }, [companyId, sendMessage]);

  useEffect(() => {
    refreshData();
  }, [refreshData]);

  return {
    ...data,
    refreshData,
    applyFilters
  };
};