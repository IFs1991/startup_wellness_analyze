export interface AnalysisResponse {
  id: string;
  type: string;
  results: any;
  created_at: string;
}

export interface AnalysisRequest {
  data: Record<string, any>[];
  target_variable?: string;
  options?: Record<string, any>;
}