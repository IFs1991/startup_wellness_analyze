"use client";

import React, { createContext, useEffect, useState, ReactNode } from 'react';
import {
  User,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  sendPasswordResetEmail,
  updateProfile,
  UserCredential
} from 'firebase/auth';
import { auth } from '@/firebase/config';

export interface AuthContextType {
  user: User | null;
  loading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<UserCredential>;
  register: (email: string, password: string, displayName: string) => Promise<void>;
  logout: () => Promise<void>;
  resetPassword: (email: string) => Promise<void>;
  updateUserProfile: (displayName: string, photoURL?: string) => Promise<void>;
}

const defaultValue: AuthContextType = {
  user: null,
  loading: true,
  error: null,
  login: () => Promise.reject('AuthContext not initialized'),
  register: () => Promise.reject('AuthContext not initialized'),
  logout: () => Promise.reject('AuthContext not initialized'),
  resetPassword: () => Promise.reject('AuthContext not initialized'),
  updateUserProfile: () => Promise.reject('AuthContext not initialized')
};

export const AuthContext = createContext<AuthContextType>(defaultValue);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const login = async (email: string, password: string): Promise<UserCredential> => {
    try {
      setError(null);
      return await signInWithEmailAndPassword(auth, email, password);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'ログインに失敗しました';
      setError(errorMessage);
      throw err;
    }
  };

  const register = async (email: string, password: string, displayName: string): Promise<void> => {
    try {
      setError(null);
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      await updateProfile(userCredential.user, { displayName });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'アカウント登録に失敗しました';
      setError(errorMessage);
      throw err;
    }
  };

  const logout = async (): Promise<void> => {
    try {
      setError(null);
      await signOut(auth);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'ログアウトに失敗しました';
      setError(errorMessage);
      throw err;
    }
  };

  const resetPassword = async (email: string): Promise<void> => {
    try {
      setError(null);
      await sendPasswordResetEmail(auth, email);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'パスワードリセットに失敗しました';
      setError(errorMessage);
      throw err;
    }
  };

  const updateUserProfile = async (displayName: string, photoURL?: string): Promise<void> => {
    try {
      setError(null);
      if (user) {
        await updateProfile(user, { displayName, photoURL: photoURL || user.photoURL });
      } else {
        throw new Error('ユーザーがログインしていません');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'プロフィール更新に失敗しました';
      setError(errorMessage);
      throw err;
    }
  };

  const value: AuthContextType = {
    user,
    loading,
    error,
    login,
    register,
    logout,
    resetPassword,
    updateUserProfile
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};