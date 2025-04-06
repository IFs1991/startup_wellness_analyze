import { useState, useCallback, useEffect } from 'react';
import { useWebSocketConnection } from './useWebSocketConnection';
import { useToast } from './use-toast';

export interface ComplianceRequirement {
  id: string;
  name: string;
  description: string;
  category: 'privacy' | 'security' | 'data' | 'operational';
  status: 'compliant' | 'non_compliant' | 'not_applicable';
  lastChecked: Date;
  nextCheck: Date;
  evidence?: {
    type: string;
    url: string;
    description: string;
  }[];
  violations?: {
    severity: 'low' | 'medium' | 'high';
    description: string;
    remediation: string;
    deadline: Date;
  }[];
}

export interface ComplianceReport {
  id: string;
  companyId: string;
  timestamp: Date;
  overallStatus: 'compliant' | 'partial' | 'non_compliant';
  requirements: ComplianceRequirement[];
  summary: {
    total: number;
    compliant: number;
    nonCompliant: number;
    notApplicable: number;
  };
  recommendations: {
    priority: 'low' | 'medium' | 'high';
    description: string;
    impact: string;
    effort: 'low' | 'medium' | 'high';
  }[];
}

export interface ComplianceConfig {
  requirements: {
    [key: string]: {
      enabled: boolean;
      checkFrequency: number; // 日数
      autoRemediation: boolean;
    };
  };
  notifications: {
    violations: boolean;
    deadlines: boolean;
    updates: boolean;
  };
  reporting: {
    format: 'pdf' | 'excel' | 'json';
    frequency: 'daily' | 'weekly' | 'monthly';
    recipients: string[];
  };
}

export const useCompliance = (companyId?: string) => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [currentReport, setCurrentReport] = useState<ComplianceReport | null>(null);
  const [config, setConfig] = useState<ComplianceConfig | null>(null);
  const [requirements, setRequirements] = useState<ComplianceRequirement[]>([]);

  const { status, sendMessage } = useWebSocketConnection('compliance');

  // コンプライアンスレポートの取得
  const fetchComplianceReport = useCallback(async () => {
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
      const response = await sendMessage('get_compliance_report', { company_id: companyId });
      setCurrentReport(response.report);
      setRequirements(response.report.requirements);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('コンプライアンスレポートの取得に失敗しました'));

      toast({
        title: 'エラー',
        description: 'コンプライアンスレポートの取得に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, companyId, toast]);

  // コンプライアンス設定の取得
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
      const response = await sendMessage('get_compliance_config', { company_id: companyId });
      setConfig(response.config);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('コンプライアンス設定の取得に失敗しました'));

      toast({
        title: 'エラー',
        description: 'コンプライアンス設定の取得に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, companyId, toast]);

  // コンプライアンス設定の更新
  const updateConfig = useCallback(async (newConfig: Partial<ComplianceConfig>): Promise<void> => {
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
      await sendMessage('update_compliance_config', {
        company_id: companyId,
        config: newConfig
      });

      // ローカルの状態を更新
      setConfig(prev => prev ? { ...prev, ...newConfig } : null);

      toast({
        title: '設定更新完了',
        description: 'コンプライアンス設定を更新しました'
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('コンプライアンス設定の更新に失敗しました'));

      toast({
        title: 'エラー',
        description: 'コンプライアンス設定の更新に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, companyId, toast]);

  // コンプライアンスチェックの実行
  const runComplianceCheck = useCallback(async (): Promise<void> => {
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
      const response = await sendMessage('run_compliance_check', { company_id: companyId });

      // 新しいレポートを設定
      setCurrentReport(response.report);
      setRequirements(response.report.requirements);

      toast({
        title: 'チェック完了',
        description: 'コンプライアンスチェックが完了しました'
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('コンプライアンスチェックの実行に失敗しました'));

      toast({
        title: 'エラー',
        description: 'コンプライアンスチェックの実行に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, companyId, toast]);

  // 違反の修正
  const remediateViolation = useCallback(async (
    requirementId: string,
    violationIndex: number,
    evidence: { type: string; url: string; description: string }
  ): Promise<void> => {
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
      const response = await sendMessage('remediate_violation', {
        company_id: companyId,
        requirement_id: requirementId,
        violation_index: violationIndex,
        evidence: evidence
      });

      // 更新されたレポートを設定
      setCurrentReport(response.report);
      setRequirements(response.report.requirements);

      toast({
        title: '修正完了',
        description: '違反の修正が完了しました'
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('違反の修正に失敗しました'));

      toast({
        title: 'エラー',
        description: '違反の修正に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, companyId, toast]);

  // 初期データの読み込み
  useEffect(() => {
    fetchComplianceReport();
    fetchConfig();
  }, [fetchComplianceReport, fetchConfig]);

  return {
    currentReport,
    config,
    requirements,
    loading,
    error,
    updateConfig,
    runComplianceCheck,
    remediateViolation,
    refreshData: () => {
      fetchComplianceReport();
      fetchConfig();
    }
  };
};