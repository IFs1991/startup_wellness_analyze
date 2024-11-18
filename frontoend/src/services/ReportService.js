import axios from 'axios';

const API_URL = '/api/report/';

const ReportService = {
  generateReport: async (reportData) => {
    try {
      const response = await axios.post(API_URL + 'generate', reportData, {
        responseType: 'blob', // バイナリデータとしてレスポンスを受け取る
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      return url;
    } catch (error) {
      throw error;
    }
  },

  getAvailableStartups: async () => {
    try {
      const response = await axios.get(API_URL + 'startups/');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getAvailableReportTypes: async () => {
    try {
      const response = await axios.get(API_URL + 'types/');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // その他のレポート関連サービス (例: 過去のレポート取得など)
  // ...
};

export default ReportService;