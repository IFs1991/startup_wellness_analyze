import React, { Suspense } from 'react';
import { createBrowserRouter, Outlet } from 'react-router-dom';
import Layout from '../components/Layout';

// レイジーロードでコンポーネントをインポート
const HomePage = React.lazy(() => import('../pages/HomePage'));
const AddStartupPage = React.lazy(() => import('../pages/AddStartupPage'));
const NotFoundPage = React.lazy(() => import('../pages/NotFoundPage'));
const CompanyDetailPage = React.lazy(() => import('../pages/CompanyDetailPage'));
const DashboardPage = React.lazy(() => import('../pages/DashboardPage'));
const CompaniesPage = React.lazy(() => import('../pages/CompaniesPage'));

// アプリのレイアウト
const AppLayout = () => {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-screen">レイアウトを読み込み中...</div>}>
      <Layout>
        <Outlet />
      </Layout>
    </Suspense>
  );
};

// ルーターの設定
export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: (
          <Suspense fallback={<div className="flex items-center justify-center h-screen">ページを読み込み中...</div>}>
            <HomePage />
          </Suspense>
        ),
      },
      {
        path: 'dashboard',
        element: (
          <Suspense fallback={<div className="flex items-center justify-center h-screen">ページを読み込み中...</div>}>
            <DashboardPage />
          </Suspense>
        ),
      },
      {
        path: 'companies',
        element: (
          <Suspense fallback={<div className="flex items-center justify-center h-screen">ページを読み込み中...</div>}>
            <CompaniesPage />
          </Suspense>
        ),
      },
      {
        path: 'startups/add',
        element: (
          <Suspense fallback={<div className="flex items-center justify-center h-screen">ページを読み込み中...</div>}>
            <AddStartupPage />
          </Suspense>
        ),
      },
      {
        path: 'companies/:id',
        element: (
          <Suspense fallback={<div className="flex items-center justify-center h-screen">ページを読み込み中...</div>}>
            <CompanyDetailPage />
          </Suspense>
        ),
      },
      {
        path: '*',
        element: (
          <Suspense fallback={<div className="flex items-center justify-center h-screen">ページを読み込み中...</div>}>
            <NotFoundPage />
          </Suspense>
        ),
      },
    ],
  },
]);