"use client";

import { useContext } from 'react';
import { AuthContext } from '../contexts/AuthContext';

/**
 * 認証情報にアクセスするためのカスタムフック
 * AuthContextからユーザー情報や認証関連の機能を提供します
 */
export const useAuth = () => {
  const context = useContext(AuthContext);

  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }

  return context;
};