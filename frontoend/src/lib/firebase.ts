import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';

// Firebaseの設定情報
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID
};

// Firebaseの初期化
const app = initializeApp(firebaseConfig);

// 認証インスタンスのエクスポート
export const auth = getAuth(app);

// 既存の認証トークン取得関数
export async function getAuthToken(): Promise<string> {
  const user = auth.currentUser;
  if (!user) {
    throw new Error('ユーザーがログインしていません');
  }
  return user.getIdToken();
}