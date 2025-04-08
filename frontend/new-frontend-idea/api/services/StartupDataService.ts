import apiClient from '../apiClient';
import { StartupData, ApiResponse } from '../types';

/**
 * スタートアップデータ関連のAPIサービス
 */
class StartupDataService {
  private static readonly BASE_PATH = '/api/startups';

  /**
   * すべてのスタートアップデータを取得する
   */
  public static async getAllStartups(): Promise<ApiResponse<StartupData[]>> {
    try {
      return await apiClient.get<ApiResponse<StartupData[]>>(
        `${this.BASE_PATH}`
      );
    } catch (error) {
      console.error('スタートアップデータ取得エラー:', error);
      throw error;
    }
  }

  /**
   * 特定のスタートアップデータを取得する
   */
  public static async getStartupById(startupId: string): Promise<ApiResponse<StartupData>> {
    try {
      return await apiClient.get<ApiResponse<StartupData>>(
        `${this.BASE_PATH}/${startupId}`
      );
    } catch (error) {
      console.error(`スタートアップデータ取得エラー (ID: ${startupId}):`, error);
      throw error;
    }
  }

  /**
   * 新しいスタートアップデータを作成する
   */
  public static async createStartup(data: Partial<StartupData>): Promise<ApiResponse<StartupData>> {
    try {
      return await apiClient.post<ApiResponse<StartupData>>(
        `${this.BASE_PATH}`,
        data
      );
    } catch (error) {
      console.error('スタートアップデータ作成エラー:', error);
      throw error;
    }
  }

  /**
   * スタートアップデータを更新する
   */
  public static async updateStartup(startupId: string, data: Partial<StartupData>): Promise<ApiResponse<StartupData>> {
    try {
      return await apiClient.put<ApiResponse<StartupData>>(
        `${this.BASE_PATH}/${startupId}`,
        data
      );
    } catch (error) {
      console.error(`スタートアップデータ更新エラー (ID: ${startupId}):`, error);
      throw error;
    }
  }

  /**
   * スタートアップデータを削除する
   */
  public static async deleteStartup(startupId: string): Promise<ApiResponse<any>> {
    try {
      return await apiClient.delete<ApiResponse<any>>(
        `${this.BASE_PATH}/${startupId}`
      );
    } catch (error) {
      console.error(`スタートアップデータ削除エラー (ID: ${startupId}):`, error);
      throw error;
    }
  }

  /**
   * スタートアップデータをCSVとしてエクスポートする
   */
  public static async exportStartupsCSV(): Promise<Blob> {
    try {
      const response = await apiClient.get<Blob>(
        `${this.BASE_PATH}/export/csv`,
        {},
        {
          responseType: 'blob',
        }
      );
      return response;
    } catch (error) {
      console.error('スタートアップデータCSVエクスポートエラー:', error);
      throw error;
    }
  }

  /**
   * CSVからスタートアップデータをインポートする
   */
  public static async importStartupsCSV(file: File): Promise<ApiResponse<any>> {
    try {
      return await apiClient.uploadFile<ApiResponse<any>>(
        `${this.BASE_PATH}/import/csv`,
        file,
        'file'
      );
    } catch (error) {
      console.error('スタートアップデータCSVインポートエラー:', error);
      throw error;
    }
  }
}

export default StartupDataService;