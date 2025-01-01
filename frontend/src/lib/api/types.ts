// API Response Types
export interface UserResponse {
  id: string;
  email: string;
  name: string;
  created_at: string;
}

export interface AnalysisResponse {
  id: string;
  type: string;
  results: any;
  created_at: string;
}

export interface VisualizationResponse {
  id: string;
  created_at: string;
  updated_at: string;
  config: Record<string, any>;
  data: Record<string, any>;
  created_by: string;
}

export interface ReportResponse {
  id: string;
  type: string;
  status: string;
  url?: string;
  created_at: string;
}