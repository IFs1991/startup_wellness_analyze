import { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import Header from './Header';
import { Sidebar } from './Sidebar';
import { Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';

/**
 * レイアウトコンポーネント
 * アプリケーションの共通レイアウトを提供します。
 * ヘッダー、サイドバー、およびコンテンツエリアを含みます。
 *
 * @returns {JSX.Element} レイアウトコンポーネント
 */
const Layout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(false);

  // 画面サイズに応じてモバイルかどうかを判定
  useEffect(() => {
    const checkScreenSize = () => {
      setIsMobile(window.innerWidth < 768);
      if (window.innerWidth < 768) {
        setSidebarOpen(false);
      } else {
        setSidebarOpen(true);
      }
    };

    // 初期チェック
    checkScreenSize();

    // リサイズイベントリスナー
    window.addEventListener('resize', checkScreenSize);
    return () => window.removeEventListener('resize', checkScreenSize);
  }, []);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      {/* サイドバー - モバイル時は条件付き表示 */}
      <div className={`shrink-0 transition-all duration-300 ${isMobile ?
        (sidebarOpen ? 'w-64' : 'w-0') : 'w-64'}`}
        style={{ overflow: 'hidden' }}>
        {sidebarOpen && <Sidebar />}
      </div>

      {/* メインコンテンツエリア */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* ヘッダー */}
        <div className="flex items-center h-16 px-4 border-b">
          {isMobile && (
            <Button
              variant="ghost"
              size="icon"
              className="mr-4"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu className="h-5 w-5" />
              <span className="sr-only">Toggle sidebar</span>
            </Button>
          )}
          <Header />
        </div>

        {/* コンテンツ */}
        <main className="flex-1 overflow-auto p-4 bg-background">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default Layout;