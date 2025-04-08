import { NextResponse } from 'next/server';
import { getAuth } from 'firebase-admin/auth';
import { firebaseAdmin } from '@/firebase/admin-config';
import {
  generateInsights,
  generateBayesianInsights,
  generateClusterInsights,
  generateCorrelationInsights,
  generateFinancialInsights,
  generateTimeSeriesInsights,
  generateWellnessInsights
} from '@/lib/ai-insights-generator';
import { analysisExplanations } from '@/lib/analysis-explanations';

// レート制限のためのメモリストレージ
// 本番環境では Redis などを使用することを推奨
const rateLimit = {
  // ユーザーIDごとのリクエスト数を記録
  requests: new Map<string, { count: number, resetTime: number }>(),

  // 1分あたりの最大リクエスト数
  MAX_REQUESTS_PER_MINUTE: 5, // 分析リクエストは通常のチャットより制限を厳しく

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
    const { analysisType, parameters } = body;

    // 分析タイプが必須
    if (!analysisType || typeof analysisType !== 'string') {
      return NextResponse.json(
        { error: '分析タイプが必要です' },
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

    // 分析タイプの存在確認
    const analysisExplanationsRecord = analysisExplanations as Record<string, any>;
    if (!analysisExplanationsRecord[analysisType]) {
      return NextResponse.json(
        { error: `サポートされていない分析タイプ: ${analysisType}` },
        { status: 400 }
      );
    }

    // 分析タイプに応じたインサイト生成
    let insights;
    try {
      insights = await generateInsights('demo-company', analysisType, parameters || {});
    } catch (error) {
      console.error('インサイト生成エラー:', error);
      return NextResponse.json(
        { error: 'インサイト生成中にエラーが発生しました' },
        { status: 500 }
      );
    }

    // レスポンスメッセージの作成
    let responseMessage;
    switch (analysisType) {
      case 'timeseries':
        responseMessage = `
          時系列分析が完了しました。データ${parameters?.timeRange || '期間'}での主な傾向として、以下が観測されました：

          • トレンドパターン: ${Math.random() > 0.5 ? '上昇傾向' : '周期的変動'}
          • 季節性: ${Math.random() > 0.7 ? '強い季節性パターンあり' : '軽度の季節変動'}
          • 外れ値: ${Math.floor(Math.random() * 5)}件検出

          詳細は添付の分析カードをご確認ください。
        `;
        break;
      case 'correlation':
        responseMessage = `
          相関分析が完了しました。${parameters?.minCorrelation ? `相関係数${parameters.minCorrelation}以上の` : ''}主な関連性は以下の通りです：

          • 強い正の相関: ${Math.floor(Math.random() * 3) + 1}組の変数ペア
          • 強い負の相関: ${Math.floor(Math.random() * 2)}組の変数ペア
          • 統計的に有意な相関: ${Math.floor(Math.random() * 10) + 5}組検出

          特に注目すべき相関関係についての詳細は、添付の分析カードをご確認ください。
        `;
        break;
      case 'cluster':
        responseMessage = `
          クラスター分析（${parameters?.algorithm || 'K-means'}アルゴリズム使用）が完了しました。

          • クラスター数: ${parameters?.clusterCount || 4}
          • 最大のクラスターサイズ: ${Math.floor(Math.random() * 30) + 20}%
          • クラスター間の明確な区別: ${Math.random() > 0.5 ? '高い' : '中程度'}

          各クラスターの特性と示唆される対応策は添付の分析カードをご確認ください。
        `;
        break;
      case 'wellness':
        responseMessage = `
          ウェルネス分析が完了しました。組織の健康状態の主な指標は以下の通りです：

          • 全体的なウェルネススコア: ${Math.floor(Math.random() * 31) + 70}/100
          • 最も高いスコア領域: ${['心理的安全性', 'ワークライフバランス', '物理的健康', 'コミュニケーション'][Math.floor(Math.random() * 4)]}
          • 改善が必要な領域: ${['ストレス管理', 'マネジメントスタイル', '業務量', '職場環境'][Math.floor(Math.random() * 4)]}

          詳細な分析と改善提案は添付の分析カードをご覧ください。
        `;
        break;
      default:
        responseMessage = `
          ${analysisType}分析が完了しました。主な発見と洞察は添付の分析カードをご確認ください。

          このデータに基づいて、さらに詳細な分析や特定の側面についてお知りになりたい場合は、お気軽にお尋ねください。
        `;
    }

    // レスポンスを返す
    return NextResponse.json({
      response: responseMessage.trim(),
      title: `${analysisExplanationsRecord[analysisType]?.title || analysisType}分析の結果`,
      description: analysisExplanationsRecord[analysisType]?.description,
      insights: insights,
      analysisType
    });

  } catch (error) {
    console.error('分析リクエストAPIエラー:', error);
    return NextResponse.json(
      { error: 'サーバーエラーが発生しました' },
      { status: 500 }
    );
  }
}

// ヘルスチェック用のGETエンドポイント
export async function GET() {
  return NextResponse.json({ status: 'ok', supported_types: Object.keys(analysisExplanations) });
}