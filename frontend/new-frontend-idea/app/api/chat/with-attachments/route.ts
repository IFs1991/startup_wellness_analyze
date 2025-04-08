import { NextRequest, NextResponse } from 'next/server';
import { verifyUserToken } from '@/firebase/admin-config';
import { generateAIResponse } from '@/lib/ai-insights-generator';
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { promises as fsPromises } from 'fs';

// 一時的なアップロードディレクトリの作成
const UPLOAD_DIR = path.join(process.cwd(), 'tmp/uploads');

// ディレクトリが存在しない場合は作成
try {
  if (!fs.existsSync(UPLOAD_DIR)) {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
  }
} catch (error) {
  console.error('アップロードディレクトリの作成に失敗:', error);
}

export async function POST(request: NextRequest) {
  try {
    // リクエストからユーザー情報を取得
    const authHeader = request.headers.get('authorization');
    const token = authHeader ? authHeader.split('Bearer ')[1] : '';
    let userId = 'anonymous';

    if (token) {
      try {
        const result = await verifyUserToken(token);
        if (result.isVerified) {
          userId = result.uid;
        } else {
          return NextResponse.json(
            { error: '認証エラー' },
            { status: 401 }
          );
        }
      } catch (error) {
        console.error('トークン検証エラー:', error);
        return NextResponse.json(
          { error: '認証エラー' },
          { status: 401 }
        );
      }
    }

    // FormDataを解析
    const formData = await request.formData();
    const message = formData.get('message') as string;

    if (!message) {
      return NextResponse.json(
        { error: 'メッセージが必要です' },
        { status: 400 }
      );
    }

    // 添付ファイルメタデータを取得
    const attachmentsMetadataStr = formData.get('attachmentsMetadata') as string;
    const attachmentsMetadata = attachmentsMetadataStr
      ? JSON.parse(attachmentsMetadataStr)
      : [];

    // ファイルの処理
    const files = formData.getAll('files') as File[];
    const savedAttachments = [];

    if (files && files.length > 0) {
      // ユーザーごとのディレクトリを作成
      const userDir = path.join(UPLOAD_DIR, userId);
      if (!fs.existsSync(userDir)) {
        fs.mkdirSync(userDir, { recursive: true });
      }

      // 各ファイルを保存
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const metadata = attachmentsMetadata[i] || {};

        // ユニークなファイル名を生成
        const fileExt = path.extname(file.name);
        const uniqueFilename = `${uuidv4()}${fileExt}`;
        const filePath = path.join(userDir, uniqueFilename);

        // ファイルを一時ディレクトリに保存
        const fileBuffer = Buffer.from(await file.arrayBuffer());
        await fsPromises.writeFile(filePath, fileBuffer);

        // 添付ファイル情報を追加
        savedAttachments.push({
          id: metadata.id || uuidv4(),
          name: file.name,
          type: file.type,
          size: file.size,
          path: filePath,
          url: `/api/attachments/${userId}/${uniqueFilename}`
        });
      }
    }

    // AIレスポンスの生成
    const insights = await generateAIResponse(message, userId);

    // レスポンス作成
    return NextResponse.json({
      response: insights.text,
      analysis: insights.analysis,
      attachments: savedAttachments.map(att => ({
        id: att.id,
        name: att.name,
        type: att.type,
        size: att.size,
        url: att.url
      }))
    });

  } catch (error) {
    console.error('APIエラー:', error);
    return NextResponse.json(
      { error: 'サーバーエラーが発生しました' },
      { status: 500 }
    );
  }
}