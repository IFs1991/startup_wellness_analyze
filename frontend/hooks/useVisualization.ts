import { useState, useCallback } from 'react';
import { useToast } from './useToast';
import VisualizationService from '../api/services/VisualizationService';
import { ApiResponse, ChartResponse, ChartRequest, DashboardSection, DashboardResponse, JobStatusResponse } from '../api/types';

/**
 * 可視化関連の状態と操作を提供するカスタムフック
 */
export const useVisualization = () => {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [chartData, setChartData] = useState<ChartResponse | null>(null);
  const [dashboardData, setDashboardData] = useState<DashboardResponse | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const { toast } = useToast();

  /**
   * エラー処理を行う共通関数
   */
  const handleError = useCallback((error: any, message: string) => {
    console.error(message, error);
    setError(error?.message || message);
    toast({
      title: 'エラー',
      description: error?.message || message,
      variant: 'destructive',
    });
    return null;
  }, [toast]);

  /**
   * 単一チャートを生成する
   */
  const generateChart = useCallback(async (
    config: any,
    data: any,
    format: string = 'png',
    templateId?: string
  ) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await VisualizationService.generateChart(config, data, format, templateId);
      setChartData(response.data as ChartResponse);
      return response.data;
    } catch (error) {
      return handleError(error, 'チャート生成中にエラーが発生しました');
    } finally {
      setIsLoading(false);
    }
  }, [handleError]);

  /**
   * 複数チャートを生成する
   */
  const generateMultipleCharts = useCallback(async (
    charts: Array<{config: any, data: any, format?: string, template_id?: string}>
  ) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await VisualizationService.generateMultipleCharts(charts);
      return response.data;
    } catch (error) {
      return handleError(error, '複数チャート生成中にエラーが発生しました');
    } finally {
      setIsLoading(false);
    }
  }, [handleError]);

  /**
   * ダッシュボードを生成する
   */
  const generateDashboard = useCallback(async (
    title: string,
    sections: Array<DashboardSection>,
    chartIds: string[],
    description?: string,
    theme: string = 'light',
    format: string = 'pdf'
  ) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await VisualizationService.generateDashboard(
        title,
        sections,
        chartIds,
        description,
        theme,
        format
      );
      setDashboardData(response.data as DashboardResponse);
      return response.data;
    } catch (error) {
      return handleError(error, 'ダッシュボード生成中にエラーが発生しました');
    } finally {
      setIsLoading(false);
    }
  }, [handleError]);

  /**
   * バックグラウンドでチャートを生成する
   */
  const generateChartBackground = useCallback(async (
    config: any,
    data: any,
    format: string = 'png',
    templateId?: string
  ) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await VisualizationService.generateChartBackground(config, data, format, templateId);
      setJobStatus(response.data as JobStatusResponse);
      return response.data;
    } catch (error) {
      return handleError(error, 'バックグラウンドチャート生成中にエラーが発生しました');
    } finally {
      setIsLoading(false);
    }
  }, [handleError]);

  /**
   * チャート生成ジョブのステータスを確認する
   */
  const getChartStatus = useCallback(async (jobId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await VisualizationService.getChartStatus(jobId);
      setJobStatus(response.data as JobStatusResponse);
      return response.data;
    } catch (error) {
      return handleError(error, 'チャートステータス確認中にエラーが発生しました');
    } finally {
      setIsLoading(false);
    }
  }, [handleError]);

  /**
   * 生成されたチャートのダウンロードURLを取得する
   */
  const getChartDownloadUrl = useCallback((chartId: string) => {
    return VisualizationService.getChartDownloadUrl(chartId);
  }, []);

  /**
   * 分析クラスの可視化を行う
   */
  const visualizeAnalyzerResults = useCallback(async (
    analyzerType: string,
    analysisResults: any,
    visualizationType: string = 'bar',
    options: any = {}
  ) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await VisualizationService.visualizeAnalyzerResults(
        analyzerType,
        analysisResults,
        visualizationType,
        options
      );
      return response.data;
    } catch (error) {
      return handleError(error, '分析結果の可視化中にエラーが発生しました');
    } finally {
      setIsLoading(false);
    }
  }, [handleError]);

  /**
   * チャートデータをリセットする
   */
  const resetChartData = useCallback(() => {
    setChartData(null);
  }, []);

  /**
   * ダッシュボードデータをリセットする
   */
  const resetDashboardData = useCallback(() => {
    setDashboardData(null);
  }, []);

  /**
   * ジョブステータスをリセットする
   */
  const resetJobStatus = useCallback(() => {
    setJobStatus(null);
  }, []);

  /**
   * すべての状態をリセットする
   */
  const resetAll = useCallback(() => {
    setChartData(null);
    setDashboardData(null);
    setJobStatus(null);
    setError(null);
  }, []);

  return {
    // 状態
    isLoading,
    error,
    chartData,
    dashboardData,
    jobStatus,

    // チャート関連の操作
    generateChart,
    generateMultipleCharts,
    generateChartBackground,
    getChartStatus,
    getChartDownloadUrl,

    // ダッシュボード関連の操作
    generateDashboard,

    // 分析可視化関連の操作
    visualizeAnalyzerResults,

    // 状態リセット関連の操作
    resetChartData,
    resetDashboardData,
    resetJobStatus,
    resetAll
  };
};

export default useVisualization;