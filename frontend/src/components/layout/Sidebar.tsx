import { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import {
  LayoutDashboard,
  Building,
  Settings,
  FileText,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

// Auth型を定義
interface AuthContextType {
  currentUser?: {
    email?: string;
    displayName?: string;
  };
}

// useAuthフックをモック
const useAuth = (): AuthContextType => {
  return {
    currentUser: {
      email: 'demo@example.com',
      displayName: 'Demo User'
    }
  };
};

// トライアル情報の型定義
interface TrialInfo {
  daysRemaining: number;
  endDate: string;
}

export function Sidebar() {
  const location = useLocation();
  const { currentUser } = useAuth();

  // トライアル情報（実際にはAPIから取得する）
  const trialInfo: TrialInfo = {
    daysRemaining: 14,
    endDate: '2023/12/31'
  };

  // メニュー項目の定義
  const menuItems = [
    { text: 'ダッシュボード', path: '/', icon: <LayoutDashboard className="h-5 w-5" /> },
    { text: '企業一覧', path: '/companies', icon: <Building className="h-5 w-5" /> },
    { text: 'レポート', path: '/reports', icon: <FileText className="h-5 w-5" /> },
    { text: '設定', path: '/settings', icon: <Settings className="h-5 w-5" /> },
  ];

  // 現在のパスがメニュー項目のパスと一致するか確認する関数
  const isActivePath = (path: string) => location.pathname === path;

  return (
    <div className="w-64 h-full border-r bg-background">
      <div className="flex items-center justify-center h-16 border-b">
        <h1 className="text-xl font-bold text-primary">Startup Wellness</h1>
      </div>

      <Separator />

      {/* トライアル情報表示エリア */}
      <Card className="m-4 p-3 bg-primary/5">
        <h3 className="text-sm font-medium mb-1">トライアル期間</h3>
        <p className="text-sm mb-2">
          残り <strong>{trialInfo.daysRemaining}日</strong> ({trialInfo.endDate}まで)
        </p>
        <Button
          variant="default"
          size="sm"
          className="w-full"
        >
          プランをアップグレード
        </Button>
      </Card>

      <ScrollArea className="h-[calc(100vh-13rem)] flex-grow">
        <div className="px-3 py-2">
          {menuItems.map((item) => (
            <Link
              key={item.text}
              to={item.path}
              className={cn(
                "flex items-center rounded-md px-3 py-2 text-sm transition-colors mb-1",
                isActivePath(item.path) ?
                  "bg-primary/10 text-primary font-medium" :
                  "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <span className={cn(
                "mr-2",
                isActivePath(item.path) ? "text-primary" : "text-muted-foreground"
              )}>
                {item.icon}
              </span>
              {item.text}
            </Link>
          ))}
        </div>
      </ScrollArea>

      {/* ユーザー情報表示エリア */}
      {currentUser && (
        <div className="absolute bottom-0 left-0 right-0 p-3 border-t">
          <div className="flex items-center">
            <div className="flex items-center justify-center w-9 h-9 rounded-full bg-primary text-primary-foreground font-bold mr-3">
              {currentUser.email?.charAt(0).toUpperCase() || 'U'}
            </div>
            <div>
              <p className="text-sm font-medium">
                {currentUser.displayName || currentUser.email}
              </p>
              <p className="text-xs text-muted-foreground">
                プロフェッショナルプラン
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}