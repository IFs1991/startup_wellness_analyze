import React, { useContext } from 'react';
import { Navigate } from 'react-router-dom';
import { AuthContext } from '../../contexts/AuthContext';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

/**
 * 認証済みユーザーのみがアクセスできるルートを提供するコンポーネント
 * 未認証の場合はログインページにリダイレクトします
 * 開発モード（import.meta.env.DEV=true）の場合は認証をバイパスします
 */
const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { user, loading } = useContext(AuthContext);

  // 開発モードの場合は認証をバイパス
  const isDevelopment = import.meta.env.DEV;

  // 認証状態の読み込み中
  if (loading) {
    return <div>読み込み中...</div>;
  }

  // 開発モードの場合は常に認証済みとして扱う
  if (isDevelopment) {
    return <>{children}</>;
  }

  // 未認証の場合はログインページへリダイレクト
  if (!user) {
    return <Navigate to="/login" replace />;
  }

  // 認証済みの場合は子コンポーネントを表示
  return <>{children}</>;
};

export default ProtectedRoute;