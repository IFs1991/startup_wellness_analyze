import { apiClient } from './client';
import type { AnalysisResponse, AnalysisRequest } from './types/analysis';

export const analysisApi = {
  getDescriptiveStats: async (data: AnalysisRequest): Promise<AnalysisResponse> => {
    const response = await apiClient.post('/api/analysis/descriptive-stats', data);
    return response.data;
  },

  getCorrelation: async (data: AnalysisRequest): Promise<AnalysisResponse> => {
    const response = await apiClient.post('/api/analysis/correlation', data);
    return response.data;
  },

  getTimeSeries: async (data: AnalysisRequest): Promise<AnalysisResponse> => {
    const response = await apiClient.post('/api/analysis/time-series', data);
    return response.data;
  },

  getClusters: async (data: AnalysisRequest): Promise<AnalysisResponse> => {
    const response = await apiClient.post('/api/analysis/cluster', data);
    return response.data;
  },

  getPCA: async (data: AnalysisRequest): Promise<AnalysisResponse> => {
    const response = await apiClient.post('/api/analysis/pca', data);
    return response.data;
  },
};