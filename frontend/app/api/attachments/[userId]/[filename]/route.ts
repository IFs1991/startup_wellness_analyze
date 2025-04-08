import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { promises as fsPromises } from 'fs';

interface RouteParams {
  params: {
    userId: string;
    filename: string;
  };
}

// 一時的なアップロードディレクトリ
const UPLOAD_DIR = path.join(process.cwd(), 'tmp/uploads');

export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { userId, filename } = params;

    // ユーザーディレクトリ内のファイルパス
    const userDir = path.join(UPLOAD_DIR, userId);
    const filePath = path.join(userDir, filename);

    // ファイルが存在するか確認
    try {
      await fsPromises.access(filePath);
    } catch (error) {
      return NextResponse.json(
        { error: 'ファイルが見つかりません' },
        { status: 404 }
      );
    }

    // ファイル情報を取得
    const fileStats = await fsPromises.stat(filePath);

    // ファイルを読み込む
    const fileBuffer = await fsPromises.readFile(filePath);

    // ファイル拡張子に基づいてMIMEタイプを推測
    const ext = path.extname(filename).toLowerCase();
    let contentType = 'application/octet-stream'; // デフォルト

    // 一般的なファイルタイプのマッピング
    const mimeTypes: Record<string, string> = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.pdf': 'application/pdf',
      '.txt': 'text/plain',
      '.csv': 'text/csv',
      '.doc': 'application/msword',
      '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      '.xls': 'application/vnd.ms-excel',
      '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      '.json': 'application/json',
    };

    if (ext in mimeTypes) {
      contentType = mimeTypes[ext];
    }

    // レスポンスヘッダーを設定
    const headers = new Headers();
    headers.set('Content-Type', contentType);
    headers.set('Content-Length', fileStats.size.toString());
    headers.set('Content-Disposition', `inline; filename="${filename}"`);

    // ファイルをレスポンスとして返す
    return new NextResponse(fileBuffer, {
      status: 200,
      headers
    });

  } catch (error) {
    console.error('ファイル取得エラー:', error);
    return NextResponse.json(
      { error: 'ファイルの取得中にエラーが発生しました' },
      { status: 500 }
    );
  }
}