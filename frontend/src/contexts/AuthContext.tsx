import React, { createContext, useState, useEffect, useContext, ReactNode } from 'react';
import {
  User,
  onAuthStateChanged,
  signOut as firebaseSignOut
} from 'firebase/auth';
import { auth } from '../firebase/config';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signOut: () => Promise<void>;
}

// デフォルト値の作成
const defaultAuthContext: AuthContextType = {
  user: null,
  loading: true,
  signOut: async () => {},
};

// Contextの作成
export const AuthContext = createContext<AuthContextType>(defaultAuthContext);

// Context Providerの作成
interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Firebaseの認証状態の監視
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user);
      setLoading(false);
    });

    // クリーンアップ関数
    return () => unsubscribe();
  }, []);

  // サインアウト処理
  const signOut = async () => {
    setLoading(true);
    try {
      await firebaseSignOut(auth);
    } catch (error) {
      console.error('サインアウト中にエラーが発生しました:', error);
    } finally {
      setLoading(false);
    }
  };

  // Context値の作成
  const value = {
    user,
    loading,
    signOut,
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading ? children : <div>Loading...</div>}
    </AuthContext.Provider>
  );
};

// カスタムフック
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};