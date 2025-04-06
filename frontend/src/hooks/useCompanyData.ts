import { useState, useEffect, useCallback } from 'react';
import { useWebSocketConnection } from './useWebSocketConnection';
import { useToast } from './use-toast';

export interface CompanyInfo {
  id: string;
  name: string;
  logo?: string;
  industry: string;
  foundingDate: string;
  location: string;
  employeeCount: number;
  fundingStage: string;
  description: string;
}

export interface EmployeeMetrics {
  wellnessScore: number;
  wellnessScoreChange: number;
  engagementRate: number;
  turnoverRate: number;
  productivityScore: number;
  satisfactionScore: number;
  mentalHealthScore: number;
  workLifeBalanceScore: number;
}

export interface AnalysisDataPoint {
  x: number;
  y: number;
  clusterId: string;
}

export interface Cluster {
  id: string;
  name: string;
  count: number;
  percentage: number;
  color: string;
  features: Record<string, any>;
  insights: string[];
}

export interface ClusterData {
  points: AnalysisDataPoint[];
  clusters: Cluster[];
  xAxisLabel: string;
  yAxisLabel: string;
}

export interface TimeSeriesData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    borderColor: string;
    backgroundColor: string;
    borderDash?: number[];
  }[];
}

export interface AssociationRule {
  antecedent: string[];
  consequent: string[];
  support: number;
  confidence: number;
  lift: number;
}

export interface TextAnalysisResult {
  sentiment: {
    positive: number;
    neutral: number;
    negative: number;
  };
  topKeywords: Array<{ word: string; count: number }>;
  topPhrases: Array<{ phrase: string; count: number }>;
  insightSummary: string;
}

export interface CompanyData {
  companyInfo?: CompanyInfo;
  employeeMetrics?: EmployeeMetrics;
  clusterData?: ClusterData;
  timeSeriesData?: TimeSeriesData;
  associationRules?: AssociationRule[];
  textAnalysis?: TextAnalysisResult;
  loading: boolean;
  error: Error | null;
}

/**
 * 企業詳細データを取得するカスタムフック
 *
 * @param companyId 企業ID
 * @returns 企業詳細データと状態
 */
export const useCompanyData = (companyId: string): CompanyData & { refreshData: () => void } => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const [companyInfo, setCompanyInfo] = useState<CompanyInfo | undefined>(undefined);
  const [employeeMetrics, setEmployeeMetrics] = useState<EmployeeMetrics | undefined>(undefined);
  const [clusterData, setClusterData] = useState<ClusterData | undefined>(undefined);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData | undefined>(undefined);
  const [associationRules, setAssociationRules] = useState<AssociationRule[] | undefined>(undefined);
  const [textAnalysis, setTextAnalysis] = useState<TextAnalysisResult | undefined>(undefined);

  // WebSocket接続を確立
  const {
    status,
    sendMessage,
    messages,
    error: wsError
  } = useWebSocketConnection(`company/${companyId}`);

  // エラー時の処理
  useEffect(() => {
    if (wsError) {
      setError(wsError);
      setLoading(false);

      toast({
        title: 'データ取得エラー',
        description: '企業データの取得に失敗しました',
        variant: 'destructive'
      });
    }
  }, [wsError, toast]);

  // 接続状態の監視
  useEffect(() => {
    if (status === 'connected') {
      // 接続成功時にデータをリクエスト
      sendMessage('get_company_data', { company_id: companyId });
    } else if (status === 'disconnected' || status === 'error') {
      setLoading(false);
    }
  }, [status, sendMessage, companyId]);

  // メッセージの処理
  useEffect(() => {
    if (messages && messages.length > 0) {
      const latestMessage = messages[messages.length - 1];

      if (latestMessage.type === 'company_data') {
        const data = latestMessage.data;

        // 各データの設定
        if (data.companyInfo) {
          setCompanyInfo(data.companyInfo);
        }

        if (data.employeeMetrics) {
          setEmployeeMetrics(data.employeeMetrics);
        }

        if (data.clusterData) {
          setClusterData(data.clusterData);
        }

        if (data.timeSeriesData) {
          setTimeSeriesData(data.timeSeriesData);
        }

        if (data.associationRules) {
          setAssociationRules(data.associationRules);
        }

        if (data.textAnalysis) {
          setTextAnalysis(data.textAnalysis);
        }

        setLoading(false);
      }
    }
  }, [messages]);

  // データの更新をリクエストする関数
  const refreshData = useCallback(() => {
    if (status === 'connected') {
      setLoading(true);
      sendMessage('get_company_data', { company_id: companyId });
    } else {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
    }
  }, [status, sendMessage, companyId, toast]);

  return {
    companyInfo,
    employeeMetrics,
    clusterData,
    timeSeriesData,
    associationRules,
    textAnalysis,
    loading,
    error,
    refreshData
  };
};