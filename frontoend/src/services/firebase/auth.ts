import React from 'react';
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  sendPasswordResetEmail,
  User
} from 'firebase/auth';
import { app } from './firebase-config';

// 認証インスタンスの取得
const auth = getAuth(app);

// ユーザー情報の型定義
export interface UserCredential {
  email: string;
  password: string;
}

export class AuthService {
  // 現在のユーザーの取得
  getCurrentUser(): User | null {
    return auth.currentUser;
  }

  // メールアドレスとパスワードでサインアップ
  async signUp({ email, password }: UserCredential): Promise<User> {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      return userCredential.user;
    } catch (error: any) {
      throw this.handleAuthError(error);
    }
  }

  // メールアドレスとパスワードでサインイン
  async signIn({ email, password }: UserCredential): Promise<User> {
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      return userCredential.user;
    } catch (error: any) {
      throw this.handleAuthError(error);
    }
  }

  // サインアウト
  async signOut(): Promise<void> {
    try {
      await signOut(auth);
    } catch (error: any) {
      throw this.handleAuthError(error);
    }
  }

  // パスワードリセットメールの送信
  async sendPasswordReset(email: string): Promise<void> {
    try {
      await sendPasswordResetEmail(auth, email);
    } catch (error: any) {
      throw this.handleAuthError(error);
    }
  }

  // 認証状態の変更を監視
  onAuthStateChange(callback: (user: User | null) => void): () => void {
    return onAuthStateChanged(auth, callback);
  }

  // エラーハンドリング
  private handleAuthError(error: any): Error {
    // Firebaseの認証エラーを日本語でわかりやすく変換
    const errorMessages: { [key: string]: string } = {
      'auth/email-already-in-use': 'このメールアドレスは既に使用されています',
      'auth/invalid-email': '無効なメールアドレスです',
      'auth/operation-not-allowed': 'この操作は許可されていません',
      'auth/weak-password': 'パスワードが脆弱です。より強力なパスワードを設定してください',
      'auth/user-disabled': 'このアカウントは無効化されています',
      'auth/user-not-found': 'ユーザーが見つかりません',
      'auth/wrong-password': 'パスワードが間違っています'
    };

    const message = errorMessages[error.code] || 'エラーが発生しました';
    return new Error(message);
  }
}

// カスタムフック用の型定義
export interface AuthState {
  user: User | null;
  loading: boolean;
  error: Error | null;
}

// 認証状態管理用のカスタムフック
export function useAuth(): AuthState {
  const [authState, setAuthState] = React.useState<AuthState>({
    user: auth.currentUser,
    loading: true,
    error: null
  });

  React.useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth,
      (user) => {
        setAuthState({ user, loading: false, error: null });
      },
      (error) => {
        setAuthState({ user: null, loading: false, error: error as Error });
      }
    );

    // クリーンアップ関数
    return () => unsubscribe();
  }, []);

  return authState;
}

// シングルトンインスタンスとしてエクスポート
const authService = new AuthService();
export default authService;