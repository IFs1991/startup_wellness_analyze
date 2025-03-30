import axios, { AxiosInstance, AxiosRequestConfig, AxiosError, AxiosResponse } from 'axios';

// 環境変数からAPIのベースURLを取得
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

/**
 * API通信のためのクライアントクラス
 */
class ApiClient {
  private api: AxiosInstance;
  private static instance: ApiClient;

  private constructor() {
    this.api = axios.create({
      baseURL: API_URL,
      timeout: 30000, // 30秒タイムアウト
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // リクエストインターセプター
    this.api.interceptors.request.use(
      (config) => {
        // トークンがあればリクエストヘッダーに追加
        const token = localStorage.getItem('auth_token');
        if (token && config.headers) {
          config.headers['Authorization'] = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        console.error('APIリクエストエラー:', error);
        return Promise.reject(error);
      }
    );

    // レスポンスインターセプター
    this.api.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        // エラーハンドリング
        if (error.response?.status === 401) {
          // 認証エラー処理
          console.error('認証エラー：再ログインが必要です');
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  /**
   * シングルトンインスタンスを取得
   */
  public static getInstance(): ApiClient {
    if (!ApiClient.instance) {
      ApiClient.instance = new ApiClient();
    }
    return ApiClient.instance;
  }

  /**
   * GETリクエストを送信
   */
  public async get<T>(url: string, params?: any, config?: AxiosRequestConfig): Promise<T> {
    try {
      const response: AxiosResponse<T> = await this.api.get(url, {
        params,
        ...config
      });
      return response.data;
    } catch (error) {
      console.error(`GET ${url} エラー:`, error);
      throw error;
    }
  }

  /**
   * POSTリクエストを送信
   */
  public async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    try {
      const response: AxiosResponse<T> = await this.api.post(url, data, config);
      return response.data;
    } catch (error) {
      console.error(`POST ${url} エラー:`, error);
      throw error;
    }
  }

  /**
   * PUTリクエストを送信
   */
  public async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    try {
      const response: AxiosResponse<T> = await this.api.put(url, data, config);
      return response.data;
    } catch (error) {
      console.error(`PUT ${url} エラー:`, error);
      throw error;
    }
  }

  /**
   * DELETEリクエストを送信
   */
  public async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    try {
      const response: AxiosResponse<T> = await this.api.delete(url, config);
      return response.data;
    } catch (error) {
      console.error(`DELETE ${url} エラー:`, error);
      throw error;
    }
  }

  /**
   * PATCHリクエストを送信
   */
  public async patch<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    try {
      const response: AxiosResponse<T> = await this.api.patch(url, data, config);
      return response.data;
    } catch (error) {
      console.error(`PATCH ${url} エラー:`, error);
      throw error;
    }
  }

  /**
   * ファイルアップロード用のPOSTリクエスト
   */
  public async uploadFile<T>(url: string, file: File, fieldName: string = 'file', additionalData?: Record<string, any>): Promise<T> {
    const formData = new FormData();
    formData.append(fieldName, file);

    // 追加データがあれば追加
    if (additionalData) {
      Object.entries(additionalData).forEach(([key, value]) => {
        formData.append(key, String(value));
      });
    }

    try {
      const response: AxiosResponse<T> = await this.api.post(url, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error(`ファイルアップロード ${url} エラー:`, error);
      throw error;
    }
  }
}

// APIクライアントのシングルトンインスタンスをエクスポート
export default ApiClient.getInstance();