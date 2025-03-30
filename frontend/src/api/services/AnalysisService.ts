import apiClient from '../apiClient';
import { AnalysisRequest, AnalysisResult, WellnessScoreResult, ApiResponse } from '../types';

/**
 * 分析関連のAPIサービス
 */
class AnalysisService {
  private static readonly BASE_PATH = '/api/analysis';

  /**
   * 新しい分析リクエストを送信する
   */
  public static async createAnalysis(request: AnalysisRequest): Promise<ApiResponse<AnalysisResult>> {
    try {
      return await apiClient.post<ApiResponse<AnalysisResult>>(
        `${this.BASE_PATH}/create`,
        request
      );
    } catch (error) {
      console.error('分析リクエスト作成エラー:', error);
      throw error;
    }
  }

  /**
   * 分析結果を取得する
   */
  public static async getAnalysisResult(analysisId: string): Promise<ApiResponse<AnalysisResult>> {
    try {
      return await apiClient.get<ApiResponse<AnalysisResult>>(
        `${this.BASE_PATH}/${analysisId}`
      );
    } catch (error) {
      console.error(`分析結果取得エラー (ID: ${analysisId}):`, error);
      throw error;
    }
  }

  /**
   * ウェルネススコアを計算する
   */
  public static async calculateWellnessScore(startupId: string): Promise<ApiResponse<WellnessScoreResult>> {
    try {
      return await apiClient.post<ApiResponse<WellnessScoreResult>>(
        `${this.BASE_PATH}/wellness-score`,
        { startup_id: startupId }
      );
    } catch (error) {
      console.error(`ウェルネススコア計算エラー (スタートアップID: ${startupId}):`, error);
      throw error;
    }
  }

  /**
   * 相関分析を実行する
   */
  public static async runCorrelationAnalysis(data: Record<string, any>): Promise<ApiResponse<any>> {
    try {
      return await apiClient.post<ApiResponse<any>>(
        `${this.BASE_PATH}/correlation`,
        {
          analysis_type: 'correlation',
          data: data
        }
      );
    } catch (error) {
      console.error('相関分析実行エラー:', error);
      throw error;
    }
  }

  /**
   * クラスター分析を実行する
   */
  public static async runClusterAnalysis(data: Record<string, any>, parameters?: Record<string, any>): Promise<ApiResponse<any>> {
    try {
      return await apiClient.post<ApiResponse<any>>(
        `${this.BASE_PATH}/cluster`,
        {
          analysis_type: 'cluster',
          data: data,
          parameters: parameters
        }
      );
    } catch (error) {
      console.error('クラスター分析実行エラー:', error);
      throw error;
    }
  }

  /**
   * 時系列分析を実行する
   */
  public static async runTimeSeriesAnalysis(data: Record<string, any>, parameters?: Record<string, any>): Promise<ApiResponse<any>> {
    try {
      return await apiClient.post<ApiResponse<any>>(
        `${this.BASE_PATH}/time-series`,
        {
          analysis_type: 'time_series',
          data: data,
          parameters: parameters
        }
      );
    } catch (error) {
      console.error('時系列分析実行エラー:', error);
      throw error;
    }
  }

  /**
   * 生存分析を実行する
   */
  public static async runSurvivalAnalysis(data: Record<string, any>): Promise<ApiResponse<any>> {
    try {
      return await apiClient.post<ApiResponse<any>>(
        `${this.BASE_PATH}/survival`,
        {
          analysis_type: 'survival',
          data: data
        }
      );
    } catch (error) {
      console.error('生存分析実行エラー:', error);
      throw error;
    }
  }

  /**
   * 主成分分析を実行する
   */
  public static async runPCAAnalysis(data: Record<string, any>): Promise<ApiResponse<any>> {
    try {
      return await apiClient.post<ApiResponse<any>>(
        `${this.BASE_PATH}/pca`,
        {
          analysis_type: 'pca',
          data: data
        }
      );
    } catch (error) {
      console.error('主成分分析実行エラー:', error);
      throw error;
    }
  }

  /**
   * すべての分析結果を取得する
   */
  public static async getAllAnalyses(): Promise<ApiResponse<AnalysisResult[]>> {
    try {
      return await apiClient.get<ApiResponse<AnalysisResult[]>>(
        `${this.BASE_PATH}/all`
      );
    } catch (error) {
      console.error('全分析結果取得エラー:', error);
      throw error;
    }
  }
}

export default AnalysisService;