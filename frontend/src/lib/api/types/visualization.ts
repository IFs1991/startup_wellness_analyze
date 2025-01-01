export interface VisualizationResponse {
  id: string;
  created_at: string;
  updated_at: string;
  config: Record<string, any>;
  data: Record<string, any>;
  created_by: string;
}

export interface CreateVisualizationRequest {
  type: 'dashboard' | 'graph';
  config: Record<string, any>;
  data?: Record<string, any>;
}

export interface UpdateVisualizationRequest {
  config?: Record<string, any>;
  data?: Record<string, any>;
}