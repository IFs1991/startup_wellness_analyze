import apiClient from '../apiClient';
import { ApiResponse, ChartResponse, ChartRequest, DashboardSection, DashboardResponse, JobStatusResponse } from '../types';

/**
 * 可視化関連のAPIサービス
 */
class VisualizationService {
  private static readonly BASE_PATH = '/api/visualizations';

  /**
   * 単一チャートを生成する
   */
  public static async generateChart(config: any, data: any, format: string = 'png', templateId?: string): Promise<ApiResponse<any>> {
    try {
      return await apiClient.post<ApiResponse<ChartResponse>>(
        `${this.BASE_PATH}/chart`,
        {
          config,
          data,
          format,
          template_id: templateId
        } as ChartRequest
      );
    } catch (error) {
      console.error('チャート生成エラー:', error);
      throw error;
    }
  }

  /**
   * 複数チャートを生成する
   */
  public static async generateMultipleCharts(charts: Array<{config: any, data: any, format?: string, template_id?: string}>): Promise<ApiResponse<any[]>> {
    try {
      return await apiClient.post<ApiResponse<ChartResponse[]>>(
        `${this.BASE_PATH}/charts`,
        { charts }
      );
    } catch (error) {
      console.error('複数チャート生成エラー:', error);
      throw error;
    }
  }

  /**
   * ダッシュボードを生成する
   */
  public static async generateDashboard(
    title: string,
    sections: Array<DashboardSection>,
    chartIds: string[],
    description?: string,
    theme: string = 'light',
    format: string = 'pdf'
  ): Promise<ApiResponse<DashboardResponse>> {
    try {
      return await apiClient.post<ApiResponse<DashboardResponse>>(
        `${this.BASE_PATH}/dashboard`,
        {
          title,
          description,
          sections,
          chart_ids: chartIds,
          theme,
          format
        }
      );
    } catch (error) {
      console.error('ダッシュボード生成エラー:', error);
      throw error;
    }
  }

  /**
   * バックグラウンドでチャートを生成する
   */
  public static async generateChartBackground(config: any, data: any, format: string = 'png', templateId?: string): Promise<ApiResponse<any>> {
    try {
      return await apiClient.post<ApiResponse<JobStatusResponse>>(
        `${this.BASE_PATH}/chart/background`,
        {
          config,
          data,
          format,
          template_id: templateId
        } as ChartRequest
      );
    } catch (error) {
      console.error('バックグラウンドチャート生成エラー:', error);
      throw error;
    }
  }

  /**
   * チャート生成ジョブのステータスを確認する
   */
  public static async getChartStatus(jobId: string): Promise<ApiResponse<any>> {
    try {
      return await apiClient.get<ApiResponse<JobStatusResponse>>(
        `${this.BASE_PATH}/status/${jobId}`
      );
    } catch (error) {
      console.error(`チャートステータス確認エラー (ジョブID: ${jobId}):`, error);
      throw error;
    }
  }

  /**
   * 生成されたチャートをダウンロードする
   */
  public static getChartDownloadUrl(chartId: string): string {
    return `${apiClient.getBaseUrl()}${this.BASE_PATH}/download/${chartId}`;
  }

  /**
   * 分析クラスの可視化を行う
   */
  public static async visualizeAnalyzerResults(
    analyzerType: string,
    analysisResults: any,
    visualizationType: string = 'bar',
    options: any = {}
  ): Promise<ApiResponse<any>> {
    try {
      return await apiClient.post<ApiResponse<any>>(
        `${this.BASE_PATH}/visualize`,
        {
          analyzer_type: analyzerType,
          analysis_results: analysisResults,
          visualization_type: visualizationType,
          options
        }
      );
    } catch (error) {
      console.error('分析結果の可視化エラー:', error);
      throw error;
    }
  }
}

export default VisualizationService;