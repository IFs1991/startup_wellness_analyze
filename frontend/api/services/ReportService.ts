import apiClient from '../apiClient';
import { Report, ApiResponse } from '../types';

/**
 * レポート関連のAPIサービス
 */
class ReportService {
  private static readonly BASE_PATH = '/api/reports';

  /**
   * 新しいレポートを作成する
   */
  public static async createReport(report: Report): Promise<ApiResponse<any>> {
    try {
      return await apiClient.post<ApiResponse<any>>(
        `${this.BASE_PATH}/create`,
        report
      );
    } catch (error) {
      console.error('レポート作成エラー:', error);
      throw error;
    }
  }

  /**
   * レポートのステータスを取得する
   */
  public static async getReportStatus(reportId: string): Promise<ApiResponse<any>> {
    try {
      return await apiClient.get<ApiResponse<any>>(
        `${this.BASE_PATH}/${reportId}/status`
      );
    } catch (error) {
      console.error(`レポートステータス取得エラー (ID: ${reportId}):`, error);
      throw error;
    }
  }

  /**
   * レポートの内容を取得する
   */
  public static async getReportContent(reportId: string): Promise<ApiResponse<any>> {
    try {
      return await apiClient.get<ApiResponse<any>>(
        `${this.BASE_PATH}/${reportId}/content`
      );
    } catch (error) {
      console.error(`レポート内容取得エラー (ID: ${reportId}):`, error);
      throw error;
    }
  }

  /**
   * レポートをPDFとしてダウンロードする
   */
  public static async downloadReportPDF(reportId: string): Promise<Blob> {
    try {
      const response = await apiClient.get<Blob>(
        `${this.BASE_PATH}/${reportId}/pdf`,
        {},
        {
          responseType: 'blob',
        }
      );
      return response;
    } catch (error) {
      console.error(`レポートPDFダウンロードエラー (ID: ${reportId}):`, error);
      throw error;
    }
  }

  /**
   * レポートテンプレートのリストを取得する
   */
  public static async getReportTemplates(): Promise<ApiResponse<any>> {
    try {
      return await apiClient.get<ApiResponse<any>>(
        `${this.BASE_PATH}/templates`
      );
    } catch (error) {
      console.error('レポートテンプレート取得エラー:', error);
      throw error;
    }
  }

  /**
   * ユーザーのレポート履歴を取得する
   */
  public static async getUserReportHistory(): Promise<ApiResponse<any>> {
    try {
      return await apiClient.get<ApiResponse<any>>(
        `${this.BASE_PATH}/history`
      );
    } catch (error) {
      console.error('レポート履歴取得エラー:', error);
      throw error;
    }
  }

  /**
   * レポートを削除する
   */
  public static async deleteReport(reportId: string): Promise<ApiResponse<any>> {
    try {
      return await apiClient.delete<ApiResponse<any>>(
        `${this.BASE_PATH}/${reportId}`
      );
    } catch (error) {
      console.error(`レポート削除エラー (ID: ${reportId}):`, error);
      throw error;
    }
  }
}

export default ReportService;