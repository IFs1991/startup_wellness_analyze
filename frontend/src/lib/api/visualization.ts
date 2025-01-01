import { apiClient } from './client';
import type { 
  VisualizationResponse, 
  CreateVisualizationRequest,
  UpdateVisualizationRequest 
} from './types/visualization';

export const visualizationApi = {
  createDashboard: async (data: CreateVisualizationRequest) => {
    const response = await apiClient.post('/visualization/dashboard', data);
    return response.data;
  },

  generateGraph: async (data: CreateVisualizationRequest) => {
    const response = await apiClient.post('/visualization/graph', data);
    return response.data;
  },

  getVisualizations: async (): Promise<VisualizationResponse[]> => {
    const response = await apiClient.get('/visualization/visualizations');
    return response.data;
  },

  getVisualization: async (id: string): Promise<VisualizationResponse> => {
    const response = await apiClient.get(`/visualization/visualization/${id}`);
    return response.data;
  },

  updateVisualization: async (
    id: string, 
    data: UpdateVisualizationRequest
  ): Promise<VisualizationResponse> => {
    const response = await apiClient.put(`/visualization/visualization/${id}`, data);
    return response.data;
  },

  deleteVisualization: async (id: string): Promise<void> => {
    await apiClient.delete(`/visualization/visualization/${id}`);
  },
};