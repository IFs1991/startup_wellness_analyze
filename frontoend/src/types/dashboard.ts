// src/types/dashboard.ts

// 既存のウィジェット設定の型定義
export interface DashboardWidgetConfig {
  id: string
  title: string
  config: {
    type: 'line' | 'bar' | 'pie'
    data: any
    options?: any
  }
}

// 新しく追加する型定義
// ダッシュボード全体の設定を表す型
export interface DashboardConfig {
  title: string
  description?: string
  // ウィジェットの配列を含むように設定
  widgets: DashboardWidgetConfig[]
}

// APIレスポンスの型定義
export interface VisualizationResponse {
  id: string
  // 既存のウィジェット設定を継承
  widget: DashboardWidgetConfig
  createdAt: string
  updatedAt: string
}