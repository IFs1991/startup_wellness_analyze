import { ReactNode, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

interface LayoutProps {
  children: ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  const menuItems = [
    { path: '/dashboard', label: 'ダッシュボード' },
    { path: '/companies', label: '企業一覧' },
    { path: '/reports', label: 'レポート' },
    { path: '/settings', label: '設定' },
  ];

  const isActivePath = (path: string) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* サイドバー - モバイルでは閉じる */}
      <div
        className={`fixed inset-y-0 left-0 z-30 w-64 bg-gray-800 shadow-lg transform ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } md:translate-x-0 md:static md:inset-auto transition-transform duration-300 ease-in-out`}
      >
        <div className="flex items-center justify-center h-16 border-b border-gray-700">
          <Link to="/" className="text-xl font-bold text-blue-400">
            スタートアップウェルネス
          </Link>
        </div>
        <nav className="mt-6">
          <ul>
            {menuItems.map((item) => (
              <li key={item.path} className="px-4 py-2">
                <Link
                  to={item.path}
                  className={`flex items-center px-4 py-2 text-sm rounded-md ${
                    isActivePath(item.path)
                      ? 'bg-blue-900 text-blue-300 font-medium'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  }`}
                >
                  <span>{item.label}</span>
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>

      {/* メインコンテンツエリア */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* ヘッダー */}
        <header className="flex items-center justify-between h-16 px-6 bg-gray-800 border-b border-gray-700">
          <button
            onClick={toggleSidebar}
            className="p-1 text-gray-400 md:hidden focus:outline-none"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M4 6h16M4 12h16M4 18h16"
              ></path>
            </svg>
          </button>
          <div className="flex items-center">
            <div className="relative">
              <button className="flex items-center text-sm text-gray-300 focus:outline-none">
                <span className="mr-2">ユーザー名</span>
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M19 9l-7 7-7-7"
                  ></path>
                </svg>
              </button>
            </div>
          </div>
        </header>

        {/* コンテンツエリア */}
        <main className="flex-1 overflow-y-auto bg-gray-900">
          {children}
        </main>
      </div>

      {/* オーバーレイ（モバイル時のみ） */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black bg-opacity-50 md:hidden"
          onClick={toggleSidebar}
        ></div>
      )}
    </div>
  );
};

export default Layout;