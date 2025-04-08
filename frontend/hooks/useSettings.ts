import { useState, useCallback, useEffect } from 'react';
import { useWebSocketConnection } from './useWebSocketConnection';
import { useToast } from './useToast';

export interface UserSettings {
  theme: 'light' | 'dark' | 'system';
  language: string;
  notifications: {
    email: boolean;
    push: boolean;
    reportGeneration: boolean;
    dataUpdates: boolean;
  };
  display: {
    showCharts: boolean;
    showTables: boolean;
    showInsights: boolean;
    defaultView: 'dashboard' | 'analysis' | 'reports';
  };
  data: {
    autoRefresh: boolean;
    refreshInterval: number;
    dataRetention: number;
    exportFormat: 'csv' | 'excel' | 'pdf';
  };
  privacy: {
    shareAnalytics: boolean;
    shareUsageData: boolean;
    marketingEmails: boolean;
  };
}

export interface SystemSettings {
  maintenance: boolean;
  version: string;
  features: {
    [key: string]: boolean;
  };
  limits: {
    maxCompanies: number;
    maxEmployees: number;
    maxReports: number;
    maxStorage: number;
  };
}

export const useSettings = () => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [userSettings, setUserSettings] = useState<UserSettings | null>(null);
  const [systemSettings, setSystemSettings] = useState<SystemSettings | null>(null);

  const { status, sendMessage } = useWebSocketConnection('settings');

  // ユーザー設定の取得
  const fetchUserSettings = useCallback(async () => {
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
      const response = await sendMessage('get_user_settings');
      setUserSettings(response.settings);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('ユーザー設定の取得に失敗しました'));

      toast({
        title: 'エラー',
        description: 'ユーザー設定の取得に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, toast]);

  // システム設定の取得
  const fetchSystemSettings = useCallback(async () => {
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
      const response = await sendMessage('get_system_settings');
      setSystemSettings(response.settings);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('システム設定の取得に失敗しました'));

      toast({
        title: 'エラー',
        description: 'システム設定の取得に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, toast]);

  // ユーザー設定の更新
  const updateUserSettings = useCallback(async (settings: Partial<UserSettings>): Promise<void> => {
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
      await sendMessage('update_user_settings', { settings });

      // ローカルの状態を更新
      setUserSettings(prev => prev ? { ...prev, ...settings } : null);

      toast({
        title: '設定更新完了',
        description: '設定を更新しました'
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('設定の更新に失敗しました'));

      toast({
        title: 'エラー',
        description: '設定の更新に失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, toast]);

  // 設定のリセット
  const resetSettings = useCallback(async (): Promise<void> => {
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
      await sendMessage('reset_user_settings');

      // デフォルト設定を取得
      await fetchUserSettings();

      toast({
        title: 'リセット完了',
        description: '設定をデフォルトに戻しました'
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('設定のリセットに失敗しました'));

      toast({
        title: 'エラー',
        description: '設定のリセットに失敗しました',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  }, [status, sendMessage, fetchUserSettings, toast]);

  // 初期データの読み込み
  useEffect(() => {
    fetchUserSettings();
    fetchSystemSettings();
  }, [fetchUserSettings, fetchSystemSettings]);

  return {
    userSettings,
    systemSettings,
    loading,
    error,
    updateUserSettings,
    resetSettings,
    refreshSettings: () => {
      fetchUserSettings();
      fetchSystemSettings();
    }
  };
};