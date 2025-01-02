import axios from 'axios';

const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const api = {
  // 分析データ取得
  analysis: {
    getVasData: (companyId: number) =>
      apiClient.get(`/api/analysis/vas_data?company_id=${companyId}`),
    getProfitLossData: (companyId: number) =>
      apiClient.get(`/api/analysis/profit_loss_data?company_id=${companyId}`),
    getDescriptiveStats: (companyId: number) =>
      apiClient.get(`/api/analysis/descriptive_stats?company_id=${companyId}`),
    getCorrelationAnalysis: (companyId: number) =>
      apiClient.get(`/api/analysis/correlation_analysis?company_id=${companyId}`),
    getTextAnalysis: (companyId: number) =>
      apiClient.get(`/api/analysis/text_analysis?company_id=${companyId}`),
    getAiSummary: (companyId: number) =>
      apiClient.get(`/api/analysis/ai_summary?company_id=${companyId}`),
  },
  // 設定関連
  settings: {
    getVasQuestions: () => apiClient.get('/api/settings/vas_questions'),
    getFreeTextItems: () => apiClient.get('/api/settings/free_text_items'),
    getProfitLossItems: () => apiClient.get('/api/settings/profit_loss_items'),
    getAiModels: () => apiClient.get('/api/settings/ai_models'),
    saveVasQuestions: (data: any) =>
      apiClient.post('/api/settings/save_vas_questions', data),
    saveFreeTextItems: (data: any) =>
      apiClient.post('/api/settings/save_free_text_items', data),
    saveAiModel: (data: any) =>
      apiClient.post('/api/settings/save_ai_model', data),
  },
};