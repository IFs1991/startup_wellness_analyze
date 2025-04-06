import { useState, useCallback, useEffect } from 'react';
// WebSocketのインポートを削除
// import { useWebSocketConnection } from './useWebSocketConnection';
import { useToast } from './use-toast';
import apiClient from '@/api/apiClient'; // APIクライアントをインポート

export interface SubscriptionPlan {
  id: string;
  name: string;
  price: number;
  features: string[];
  maxCompanies: number;
  maxEmployees: number;
  maxReports: number;
  maxStorage: number;
  supportLevel: 'basic' | 'standard' | 'premium';
}

export interface SubscriptionStatus {
  planId: string;
  status: 'active' | 'cancelled' | 'expired' | 'pending';
  startDate: Date;
  endDate: Date;
  autoRenew: boolean;
  paymentMethod?: {
    type: string;
    last4: string;
    expiryDate: string;
  };
  usage: {
    companies: number;
    employees: number;
    reports: number;
    storage: number;
  };
}

export const useSubscription = () => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [currentPlan, setCurrentPlan] = useState<SubscriptionPlan | null>(null);
  const [subscriptionStatus, setSubscriptionStatus] = useState<SubscriptionStatus | null>(null);
  const [availablePlans, setAvailablePlans] = useState<SubscriptionPlan[]>([]);

  // WebSocket接続関連のコードを削除
  // const { status, sendMessage } = useWebSocketConnection('subscription');

  // 利用可能なプランを取得 (HTTP APIを使用)
  const fetchAvailablePlans = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response: any = await apiClient.get('/api/subscription/plans');
      // APIレスポンスの形式に合わせて調整が必要
      const formattedPlans = response.data.plans.map((plan: any) => ({
        id: plan.id,
        name: plan.name,
        price: plan.price,
        features: plan.features,
        maxCompanies: plan.maxCompanies,
        maxEmployees: plan.maxEmployees,
        maxReports: plan.metadata?.max_reports ? parseInt(plan.metadata.max_reports) : 10, // 仮の値
        maxStorage: plan.metadata?.max_storage ? parseInt(plan.metadata.max_storage) : 1024, // 仮の値 (MB)
        supportLevel: plan.metadata?.support_level || 'basic', // 仮の値
      }));
      setAvailablePlans(formattedPlans);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('プラン情報の取得に失敗しました'));
      toast({
        title: 'エラー',
        description: 'プラン情報の取得に失敗しました',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  // 現在のサブスクリプション状態を取得 (HTTP APIを使用)
  const fetchSubscriptionStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response: any = await apiClient.get('/api/subscription/status');
      // APIレスポンスの形式に合わせて調整が必要
      const statusData = response.data.status;
      if (statusData) {
        setSubscriptionStatus({
          planId: statusData.plan_id,
          status: statusData.status,
          startDate: new Date(statusData.current_period_start * 1000),
          endDate: new Date(statusData.current_period_end * 1000),
          autoRenew: !statusData.cancel_at_period_end,
          paymentMethod: statusData.payment_method ? {
            type: statusData.payment_method.card?.brand || 'unknown',
            last4: statusData.payment_method.card?.last4 || '****',
            expiryDate: `${statusData.payment_method.card?.exp_month}/${statusData.payment_method.card?.exp_year}`,
          } : undefined,
          usage: statusData.usage || { companies: 0, employees: 0, reports: 0, storage: 0 }, // 仮のusage
        });

        // 現在のプラン情報を設定 (fetchAvailablePlans が完了している前提)
        const plan = availablePlans.find(p => p.id === statusData.plan_id);
        if (plan) {
          setCurrentPlan(plan);
        }
      } else {
        setSubscriptionStatus(null);
        setCurrentPlan(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error('サブスクリプション状態の取得に失敗しました'));
      toast({
        title: 'エラー',
        description: 'サブスクリプション状態の取得に失敗しました',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }, [availablePlans, toast]);

  // プランの変更 (HTTP APIを使用)
  const changePlan = useCallback(async (planId: string, paymentMethodId?: string): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      await apiClient.post('/api/subscription/change', { plan_id: planId, payment_method_id: paymentMethodId });
      toast({
        title: 'プラン変更リクエスト完了',
        description: 'プランの変更処理を開始しました。状態が更新されるまでお待ちください。',
      });
      // 状態を即時更新 (反映には時間がかかる場合がある)
      await fetchSubscriptionStatus();
    } catch (err: any) {
      const errorDetail = err.response?.data?.detail || 'プランの変更に失敗しました';
      setError(err instanceof Error ? err : new Error(errorDetail));
      toast({
        title: 'エラー',
        description: errorDetail,
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }, [fetchSubscriptionStatus, toast]);

  // サブスクリプションのキャンセル (HTTP APIを使用)
  const cancelSubscription = useCallback(async (atPeriodEnd: boolean = true): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      await apiClient.post('/api/subscription/cancel', { at_period_end: atPeriodEnd });
      toast({
        title: 'キャンセルリクエスト完了',
        description: 'サブスクリプションのキャンセル処理を開始しました。',
      });
      // 状態を更新
      await fetchSubscriptionStatus();
    } catch (err: any) {
      const errorDetail = err.response?.data?.detail || 'サブスクリプションのキャンセルに失敗しました';
      setError(err instanceof Error ? err : new Error(errorDetail));
      toast({
        title: 'エラー',
        description: errorDetail,
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }, [fetchSubscriptionStatus, toast]);

  // 自動更新の設定 - この機能は main.py にまだ実装されていないためコメントアウト
  /*
  const setAutoRenew = useCallback(async (autoRenew: boolean): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      // TODO: バックエンドに /api/subscription/set-autorenew エンドポイントを実装する
      await apiClient.post('/api/subscription/set-autorenew', { auto_renew: autoRenew });
      toast({
        title: '設定完了',
        description: `自動更新を${autoRenew ? '有効' : '無効'}にしました`,
      });
      await fetchSubscriptionStatus();
    } catch (err) {
      setError(err instanceof Error ? err : new Error('自動更新の設定に失敗しました'));
      toast({
        title: 'エラー',
        description: '自動更新の設定に失敗しました',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  }, [fetchSubscriptionStatus, toast]);
  */

  // 初期データの読み込み
  useEffect(() => {
    fetchAvailablePlans();
  }, [fetchAvailablePlans]);

  // availablePlansが読み込まれた後にステータスを取得
  useEffect(() => {
    if (availablePlans.length > 0) {
      fetchSubscriptionStatus();
    }
  }, [availablePlans, fetchSubscriptionStatus]);

  return {
    currentPlan,
    subscriptionStatus,
    availablePlans,
    loading,
    error,
    changePlan,
    cancelSubscription,
    // setAutoRenew, // 未実装のためコメントアウト
    refreshStatus: fetchSubscriptionStatus,
  };
};