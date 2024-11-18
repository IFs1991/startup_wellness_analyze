import axios from 'axios';

const API_URL = '/api/analysis/';

const AnalysisService = {
  getAvailableStartups: async () => {
    try {
      const response = await axios.get(API_URL + 'startups/');
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  getGoogleFormQuestions: async () => {
    try {
      const response = await axios.get(API_URL + 'google_forms/questions'); // APIエンドポイント
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  getFinancialDataItems: async () => {
    try {
      const response = await axios.get(API_URL + 'financial_data/items');
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  getAnalysisMethods: async () => {
    try {
      const response = await axios.get(API_URL + 'methods');
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  runAnalysis: async (startupId, analysisType) => {
    try {
      const response = await axios.post(API_URL + 'run', { startupId, analysisType });
      return response.data;
    } catch (error) {
      throw error;
    }
  },

  saveAnalysisSettings: async (settings) => {
    try {
      const response = await axios.post(API_URL + 'settings', settings);
      return response.data;
    } catch (error) {
      throw error;
    }
  },

};

export default AnalysisService;