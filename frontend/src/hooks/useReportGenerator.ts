import { useState, useCallback } from 'react';
import { useWebSocketConnection } from './useWebSocketConnection';
import { useToast } from './use-toast';

export interface ReportConfig {
  title: string;
  description?: string;
  reportType: 'pdf' | 'custom';
  parameters?: Record<string, any>;
}

export interface ReportStatus {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress?: number;
  error?: string;
  url?: string;
  createdAt: Date;
  updatedAt: Date;
}

export const useReportGenerator = (companyId?: string) => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [currentReport, setCurrentReport] = useState<ReportStatus | null>(null);

  const { status, sendMessage } = useWebSocketConnection('reports');

  const generateReport = useCallback(async (config: ReportConfig): Promise<void> => {
    if (status !== 'connected') {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const reportId = Date.now().toString();
      const initialStatus: ReportStatus = {
        id: reportId,
        status: 'pending',
        createdAt: new Date(),
        updatedAt: new Date()
      };

      setCurrentReport(initialStatus);

      const requestPayload = {
        report_id: reportId,
        company_id: companyId,
        config: {
          ...config,
          report_type: config.reportType
        }
      };

      sendMessage('generate_report', requestPayload);

      // レポート生成の進捗を監視
      const checkStatus = setInterval(async () => {
        try {
          const response = await sendMessage('check_report_status', { report_id: reportId });

          if (response.status === 'completed') {
            clearInterval(checkStatus);
            setCurrentReport({
              ...response,
              updatedAt: new Date()
            });
            setLoading(false);

            toast({
              title: 'レポート生成完了',
              description: 'レポートの生成が完了しました',
            });
          } else if (response.status === 'error') {
            clearInterval(checkStatus);
            setError(new Error(response.error || 'レポート生成に失敗しました'));
            setLoading(false);

            toast({
              title: 'エラー',
              description: response.error || 'レポート生成に失敗しました',
              variant: 'destructive'
            });
          } else {
            setCurrentReport({
              ...response,
              updatedAt: new Date()
            });
          }
        } catch (err) {
          clearInterval(checkStatus);
          setError(err instanceof Error ? err : new Error('レポート状態の確認に失敗しました'));
          setLoading(false);
        }
      }, 5000); // 5秒ごとに状態を確認

      // 30秒後にタイムアウト
      setTimeout(() => {
        clearInterval(checkStatus);
        if (loading) {
          setError(new Error('レポート生成がタイムアウトしました'));
          setLoading(false);

          toast({
            title: 'タイムアウト',
            description: 'レポート生成に時間がかかりすぎています',
            variant: 'destructive'
          });
        }
      }, 30000);

    } catch (err) {
      setError(err instanceof Error ? err : new Error('レポート生成リクエストに失敗しました'));
      setLoading(false);

      toast({
        title: 'エラー',
        description: 'レポート生成リクエストに失敗しました',
        variant: 'destructive'
      });
    }
  }, [status, sendMessage, companyId, toast]);

  const cancelReport = useCallback(async (reportId: string): Promise<void> => {
    if (status !== 'connected') {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
      return;
    }

    try {
      await sendMessage('cancel_report', { report_id: reportId });
      setCurrentReport(null);

      toast({
        title: 'キャンセル完了',
        description: 'レポート生成をキャンセルしました'
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('レポートのキャンセルに失敗しました'));

      toast({
        title: 'エラー',
        description: 'レポートのキャンセルに失敗しました',
        variant: 'destructive'
      });
    }
  }, [status, sendMessage, toast]);

  return {
    generateReport,
    cancelReport,
    currentReport,
    loading,
    error
  };
};