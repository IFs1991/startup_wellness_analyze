import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

// Firebaseの設定
// 注意: 実際のプロジェクトでは環境変数を使用することをお勧めします
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || "your-api-key",
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || "your-auth-domain",
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || "your-project-id",
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || "your-storage-bucket",
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || "your-messaging-sender-id",
  appId: import.meta.env.VITE_FIREBASE_APP_ID || "your-app-id"
};

// Firebaseアプリの初期化
const app = initializeApp(firebaseConfig);

// 認証サービスのインスタンスを取得
export const auth = getAuth(app);

// Firestoreのインスタンスを取得
export const db = getFirestore(app);

export default app;