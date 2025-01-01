import axios from 'axios';

const baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
});

// レスポンスインターセプター
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // エラーハンドリング
    if (error.response?.status === 401) {
      // 認証エラー時の処理
      console.error('Authentication error');
    }
    return Promise.reject(error);
  }
);