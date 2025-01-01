import { Home, BarChart2, FileText, Settings } from 'lucide-react';

export const routes = [
  { path: '/', label: 'ホーム', icon: Home },
  { path: '/analysis', label: '分析', icon: BarChart2 },
  { path: '/reports', label: 'レポート', icon: FileText },
  { path: '/settings', label: '設定', icon: Settings },
] as const;