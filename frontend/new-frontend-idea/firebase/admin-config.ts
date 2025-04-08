import { initializeApp, getApps, cert } from 'firebase-admin/app';

// サービスアカウント認証情報（環境変数または安全な場所に保管されるべき）
const serviceAccount = process.env.FIREBASE_SERVICE_ACCOUNT_KEY
  ? JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_KEY)
  : {
      // 開発環境用のダミーデータ（実際のプロジェクトでは必ず環境変数を使用すること）
      projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || 'dummy-project',
      clientEmail: 'dummy@example.com',
      privateKey: 'dummy-key',
    };

// Firebase Admin SDKの初期化（Nextのサーバーコンポーネントで複数回読み込まれる可能性があるため、既存のアプリを再利用）
export const firebaseAdmin = getApps().length === 0
  ? initializeApp({
      credential: cert(serviceAccount),
    })
  : getApps()[0];

// Firebase Adminを使用した検証用の便利な関数
export async function verifyUserToken(token: string) {
  const { getAuth } = await import('firebase-admin/auth');
  try {
    const auth = getAuth(firebaseAdmin);
    const decodedToken = await auth.verifyIdToken(token);
    return {
      uid: decodedToken.uid,
      email: decodedToken.email,
      isVerified: true
    };
  } catch (error) {
    console.error('トークン検証エラー:', error);
    return { isVerified: false, error };
  }
}