export interface DashboardWidgetConfig {
    id: string
    title: string
    config: {
      type: 'line' | 'bar' | 'pie'
      data: any
      options?: any
    }
  }