import axios from 'axios';

const API_BASE_URL = '/api/'; // ベースURLを定義

const api = axios.create({
  baseURL: API_BASE_URL,
});

// リクエストインターセプター (JWT トークン追加など)
api.interceptors.request.use((config) => {
  const user = JSON.parse(localStorage.getItem('user'));
  if (user && user.access_token) {
    config.headers['Authorization'] = 'Bearer ' + user.access_token;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});


// レスポンスインターセプター (エラー処理など)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // エラー処理 (例: 認証エラー時にログアウトなど)
    if (error.response.status === 401) {
      AuthService.logout();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;