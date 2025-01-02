import { Home, Building2, FileText, Settings } from 'lucide-react';

export const routes = [
  { 
    path: '/', 
    label: 'ダッシュボード', 
    icon: Home 
  },
  { 
    path: '/companies', 
    label: '企業一覧', 
    icon: Building2,
    badge: '23'
  },
  { 
    path: '/reports', 
    label: 'レポート', 
    icon: FileText,
    badge: '新規'
  },
  { 
    path: '/settings', 
    label: '設定', 
    icon: Settings 
  },
] as const;