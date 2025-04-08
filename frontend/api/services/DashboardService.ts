import apiClient from '../apiClient';
import { DashboardConfig, ApiResponse, VisualizationResponse } from '../types';

/**
 * ダッシュボード関連のAPIサービス
 */
class DashboardService {
  private static readonly BASE_PATH = '/api/dashboard';

  /**
   * ダッシュボード設定を取得する
   */
  public static async getDashboardConfig(): Promise<ApiResponse<DashboardConfig>> {
    try {
      return await apiClient.get<ApiResponse<DashboardConfig>>(
        `${this.BASE_PATH}/config`
      );
    } catch (error) {
      console.error('ダッシュボード設定取得エラー:', error);
      throw error;
    }
  }

  /**
   * ダッシュボード設定を更新する
   */
  public static async updateDashboardConfig(config: DashboardConfig): Promise<ApiResponse<DashboardConfig>> {
    try {
      return await apiClient.put<ApiResponse<DashboardConfig>>(
        `${this.BASE_PATH}/config`,
        config
      );
    } catch (error) {
      console.error('ダッシュボード設定更新エラー:', error);
      throw error;
    }
  }

  /**
   * ダッシュボード用のグラフデータを取得する
   */
  public static async getGraphData(graphId: string): Promise<ApiResponse<VisualizationResponse>> {
    try {
      return await apiClient.get<ApiResponse<VisualizationResponse>>(
        `${this.BASE_PATH}/graph/${graphId}`
      );
    } catch (error) {
      console.error(`グラフデータ取得エラー (ID: ${graphId}):`, error);
      throw error;
    }
  }

  /**
   * ダッシュボード用の複数グラフデータを一括取得する
   */
  public static async getBatchGraphData(graphIds: string[]): Promise<ApiResponse<Record<string, VisualizationResponse>>> {
    try {
      return await apiClient.post<ApiResponse<Record<string, VisualizationResponse>>>(
        `${this.BASE_PATH}/graphs/batch`,
        { graph_ids: graphIds }
      );
    } catch (error) {
      console.error('一括グラフデータ取得エラー:', error);
      throw error;
    }
  }

  /**
   * ダッシュボードの概要データを取得する
   */
  public static async getDashboardSummary(): Promise<ApiResponse<any>> {
    try {
      return await apiClient.get<ApiResponse<any>>(
        `${this.BASE_PATH}/summary`
      );
    } catch (error) {
      console.error('ダッシュボード概要取得エラー:', error);
      throw error;
    }
  }

  /**
   * ダッシュボードウィジェットを作成する
   */
  public static async createWidget(widgetData: any): Promise<ApiResponse<any>> {
    try {
      return await apiClient.post<ApiResponse<any>>(
        `${this.BASE_PATH}/widget`,
        widgetData
      );
    } catch (error) {
      console.error('ウィジェット作成エラー:', error);
      throw error;
    }
  }

  /**
   * ダッシュボードウィジェットを更新する
   */
  public static async updateWidget(widgetId: string, widgetData: any): Promise<ApiResponse<any>> {
    try {
      return await apiClient.put<ApiResponse<any>>(
        `${this.BASE_PATH}/widget/${widgetId}`,
        widgetData
      );
    } catch (error) {
      console.error(`ウィジェット更新エラー (ID: ${widgetId}):`, error);
      throw error;
    }
  }

  /**
   * ダッシュボードウィジェットを削除する
   */
  public static async deleteWidget(widgetId: string): Promise<ApiResponse<any>> {
    try {
      return await apiClient.delete<ApiResponse<any>>(
        `${this.BASE_PATH}/widget/${widgetId}`
      );
    } catch (error) {
      console.error(`ウィジェット削除エラー (ID: ${widgetId}):`, error);
      throw error;
    }
  }

  /**
   * ダッシュボードレイアウトを更新する
   */
  public static async updateLayout(layout: Record<string, any>): Promise<ApiResponse<any>> {
    try {
      return await apiClient.put<ApiResponse<any>>(
        `${this.BASE_PATH}/layout`,
        { layout }
      );
    } catch (error) {
      console.error('レイアウト更新エラー:', error);
      throw error;
    }
  }
}

export default DashboardService;