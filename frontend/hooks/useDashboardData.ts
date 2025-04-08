"use client";

import { useState, useEffect, useCallback } from 'react';
import { useWebSocketConnection } from './useWebSocketConnection';
import { useAuth } from './useAuth';
import { useToast } from './useToast';

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
export const useDashboardData = () => {
  const { user } = useAuth();
  const { toast } = useToast();
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { status, messages, error: wsError, sendMessage } = useWebSocketConnection('dashboard');

  // WebSocketエラーの処理
  useEffect(() => {
    if (wsError) {
      setError(wsError.message);
      setLoading(false);
      toast({
        title: 'エラー',
        description: 'ダッシュボードデータの取得に失敗しました',
        variant: 'destructive'
      });
    }
  }, [wsError, toast]);

  // メッセージの処理
  useEffect(() => {
    if (messages && messages.length > 0) {
      const latestMessage = messages[messages.length - 1];
      if (latestMessage.type === 'dashboard_data') {
        setData(latestMessage.data);
        setLoading(false);
        setError(null);
      }
    }
  }, [messages]);

  // 接続状態の監視
  useEffect(() => {
    if (status === 'connected') {
      refreshData();
    }
  }, [status]);

  const refreshData = useCallback(() => {
    setLoading(true);
    sendMessage('get_dashboard_data');
  }, [sendMessage]);

  const getDataForPeriod = useCallback((period: string) => {
    setLoading(true);
    sendMessage('get_dashboard_data', { period });
  }, [sendMessage]);

  useEffect(() => {
    if (user) {
      refreshData();
    }
  }, [user, refreshData]);

  return {
    data,
    loading,
    error,
    refreshData,
    getDataForPeriod
  };
};