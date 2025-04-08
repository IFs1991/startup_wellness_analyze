import { NextResponse } from 'next/server';
import { getAuth } from 'firebase-admin/auth';
import { firebaseAdmin } from '@/firebase/admin-config';
import { generateAIResponse } from '@/lib/ai-insights-generator';

// レート制限のための単純なメモリストレージ
// 本番環境では Redis などを使用することを推奨
const rateLimit = {
  // ユーザーIDごとのリクエスト数を記録
  requests: new Map<string, { count: number, resetTime: number }>(),

  // 1分あたりの最大リクエスト数
  MAX_REQUESTS_PER_MINUTE: 10,

  // レート制限をチェック
  check: (userId: string): boolean => {
    const now = Date.now();
    const userRequests = rateLimit.requests.get(userId);

    // 新しいユーザーまたはリセット時間を過ぎた場合
    if (!userRequests || userRequests.resetTime < now) {
      rateLimit.requests.set(userId, {
        count: 1,
        resetTime: now + 60000 // 1分後
      });
      return true;
    }

    // リクエスト数が上限に達した場合
    if (userRequests.count >= rateLimit.MAX_REQUESTS_PER_MINUTE) {
      return false;
    }

    // リクエスト数を増やす
    userRequests.count += 1;
    rateLimit.requests.set(userId, userRequests);
    return true;
  }
};

export async function POST(request: Request) {
  try {
    // リクエストデータを取得
    const body = await request.json();
    const { message } = body;

    // メッセージが空の場合はエラー
    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'メッセージが必要です' },
        { status: 400 }
      );
    }

    // ユーザー認証
    const authToken = request.headers.get('authorization')?.split('Bearer ')[1];
    let userId = 'anonymous';

    if (authToken) {
      try {
        // Firebase Admin SDKを使用してトークンを検証
        const auth = getAuth(firebaseAdmin);
        const decodedToken = await auth.verifyIdToken(authToken);
        userId = decodedToken.uid;
      } catch (authError) {
        console.error('認証エラー:', authError);
        return NextResponse.json(
          { error: '認証エラー' },
          { status: 401 }
        );
      }
    }

    // レート制限のチェック
    if (!rateLimit.check(userId)) {
      return NextResponse.json(
        { error: 'レート制限に達しました。しばらく待ってから再試行してください。' },
        { status: 429 }
      );
    }

    // AIレスポンスを生成
    const aiResponse = await generateAIResponse(message, userId);

    // レスポンスを返す
    return NextResponse.json({
      response: aiResponse.text,
      analysis: aiResponse.analysis
    });

  } catch (error) {
    console.error('チャットAPIエラー:', error);
    return NextResponse.json(
      { error: 'サーバーエラーが発生しました' },
      { status: 500 }
    );
  }
}

// ヘルスチェック用のGETエンドポイント
export async function GET() {
  return NextResponse.json({ status: 'ok' });
}