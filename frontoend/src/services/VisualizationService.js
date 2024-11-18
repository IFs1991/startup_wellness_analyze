import axios from 'axios';

const API_URL = '/api/visualization/';

const VisualizationService = {
  getChartData: async () => {
    try {
      const response = await axios.get(API_URL + 'chart_data/'); // 適切なエンドポイントを設定
      return response.data;
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

  getAvailableVisualizationTypes: async () => {
    try {
      const response = await axios.get(API_URL + 'types/');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  getVisualizationData: async (params) => {
    try {
      const response = await axios.get(API_URL + 'data', { params });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  // その他の可視化関連のサービス関数 (例: グラフ設定の保存など)
  // ...
};

export default VisualizationService;