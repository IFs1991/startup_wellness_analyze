// APIレスポンスの基本型
export interface ApiResponse<T> {
  data?: T;
  status: 'success' | 'error';
  message?: string;
}

// ヘルスチェックのレスポンス型
export interface HealthCheckResponse {
  status: string;
  timestamp: string;
  version: string;
  services: {
    database: string;
    redis: string;
    firebase: string;
  };
}

// ダッシュボード設定の型
export interface DashboardConfig {
  title: string;
  description?: string;
  widgets: Widget[];
  layout: Record<string, any>;
}

// ウィジェットの型
export interface Widget {
  id: string;
  type: string;
  title: string;
  data_source: string;
  settings: Record<string, any>;
  filters?: Record<string, any>[];
}

// グラフ設定の型
export interface GraphConfig {
  type: string;
  title: string;
  data_source: string;
  settings: Record<string, any>;
  filters?: Record<string, any>[];
}

// 可視化レスポンスの型
export interface VisualizationResponse {
  id: string;
  created_at: string;
  updated_at: string;
  config: Record<string, any>;
  data: Record<string, any>;
  created_by: string;
}

// レポートの型
export interface Report {
  title: string;
  description?: string;
  report_type: string;
  parameters?: Record<string, any>;
}

// スタートアップデータの型
export interface StartupData {
  id: string;
  name: string;
  founded_date: string;
  industry: string;
  location: string;
  funding_stage: string;
  total_funding: number;
  employees_count: number;
  revenue?: number;
  burn_rate?: number;
  runway?: number;
  [key: string]: any; // その他のプロパティを許容
}

// ウェルネススコア分析結果の型
export interface WellnessScoreResult {
  startup_id: string;
  startup_name: string;
  overall_score: number;
  financial_health: number;
  team_health: number;
  market_health: number;
  product_health: number;
  analysis_date: string;
  detailed_scores: Record<string, number>;
  recommendations: string[];
}

// 分析リクエストの型
export interface AnalysisRequest {
  analysis_type: string;
  data: Record<string, any>;
  parameters?: Record<string, any>;
}

// 分析結果の型
export interface AnalysisResult {
  id: string;
  status: 'pending' | 'completed' | 'error';
  data?: Record<string, any>;
  error?: string;
  last_updated: string;
}

// ユーザープロフィールの型
export interface UserProfile {
  uid: string;
  email: string;
  displayName?: string;
  role: string;
  organization?: string;
  created_at: string;
  last_login?: string;
}