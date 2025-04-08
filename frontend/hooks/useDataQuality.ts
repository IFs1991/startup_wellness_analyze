import { useState, useCallback, useEffect } from 'react';
import { useWebSocketConnection } from './useWebSocketConnection';
import { useToast } from './useToast';

export interface DataQualityMetric {
  name: string;
  value: number;
  threshold: number;
  status: 'good' | 'warning' | 'critical';
  description: string;
  lastUpdated: Date;
}

export interface DataQualityReport {
  id: string;
  companyId: string;
  timestamp: Date;
  overallScore: number;
  metrics: DataQualityMetric[];
  issues: {
    type: 'missing' | 'invalid' | 'inconsistent' | 'outdated';
    severity: 'low' | 'medium' | 'high';
    description: string;
    affectedFields: string[];
    recommendations: string[];
  }[];
  recommendations: {
    priority: 'low' | 'medium' | 'high';
    description: string;
    impact: string;
    effort: 'low' | 'medium' | 'high';
  }[];
}

export interface DataQualityConfig {
  thresholds: {
    [key: string]: number;
  };
  rules: {
    [key: string]: {
      enabled: boolean;
      parameters: Record<string, any>;
    };
  };
  autoFix: {
    enabled: boolean;
    rules: string[];
  };
}

export const useDataQuality = (companyId?: string) => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [currentReport, setCurrentReport] = useState<DataQualityReport | null>(null);
  const [config, setConfig] = useState<DataQualityConfig | null>(null);
  const [metrics, setMetrics] = useState<DataQualityMetric[]>([]);

  const { status, sendMessage } = useWebSocketConnection('data_quality');

  // データ品質レポートの取得
  const fetchQualityReport = useCallback(async () => {
    if (status !== 'connected') {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
      return;
    }

    setLoading(true);
    try {
      const response = await sendMessage('get_quality_report', { company_id: companyId });
      setCurrentReport(response.report);
      setMetrics(response.report.metrics);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('データ品質レポートの取得に失敗しました'));

      toast({
        title: 'エラー',
        description: 'データ品質レポートの取得に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, companyId, toast]);

  // データ品質設定の取得
  const fetchConfig = useCallback(async () => {
    if (status !== 'connected') {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
      return;
    }

    setLoading(true);
    try {
      const response = await sendMessage('get_quality_config', { company_id: companyId });
      setConfig(response.config);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('データ品質設定の取得に失敗しました'));

      toast({
        title: 'エラー',
        description: 'データ品質設定の取得に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, companyId, toast]);

  // データ品質設定の更新
  const updateConfig = useCallback(async (newConfig: Partial<DataQualityConfig>): Promise<void> => {
    if (status !== 'connected') {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
      return;
    }

    setLoading(true);
    try {
      await sendMessage('update_quality_config', {
        company_id: companyId,
        config: newConfig
      });

      // ローカルの状態を更新
      setConfig(prev => prev ? { ...prev, ...newConfig } : null);

      toast({
        title: '設定更新完了',
        description: 'データ品質設定を更新しました'
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('データ品質設定の更新に失敗しました'));

      toast({
        title: 'エラー',
        description: 'データ品質設定の更新に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, companyId, toast]);

  // データ品質チェックの実行
  const runQualityCheck = useCallback(async (): Promise<void> => {
    if (status !== 'connected') {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
      return;
    }

    setLoading(true);
    try {
      const response = await sendMessage('run_quality_check', { company_id: companyId });

      // 新しいレポートを設定
      setCurrentReport(response.report);
      setMetrics(response.report.metrics);

      toast({
        title: 'チェック完了',
        description: 'データ品質チェックが完了しました'
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('データ品質チェックの実行に失敗しました'));

      toast({
        title: 'エラー',
        description: 'データ品質チェックの実行に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, companyId, toast]);

  // 自動修正の実行
  const runAutoFix = useCallback(async (): Promise<void> => {
    if (status !== 'connected') {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
      return;
    }

    setLoading(true);
    try {
      const response = await sendMessage('run_auto_fix', { company_id: companyId });

      // 修正後のレポートを設定
      setCurrentReport(response.report);
      setMetrics(response.report.metrics);

      toast({
        title: '修正完了',
        description: 'データの自動修正が完了しました'
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('データの自動修正に失敗しました'));

      toast({
        title: 'エラー',
        description: 'データの自動修正に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, companyId, toast]);

  // 初期データの読み込み
  useEffect(() => {
    fetchQualityReport();
    fetchConfig();
  }, [fetchQualityReport, fetchConfig]);

  return {
    currentReport,
    config,
    metrics,
    loading,
    error,
    updateConfig,
    runQualityCheck,
    runAutoFix,
    refreshData: () => {
      fetchQualityReport();
      fetchConfig();
    }
  };
};