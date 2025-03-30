/**
 * HTMLからPDFへ変換するPuppeteerを使用したコンバーター
 *
 * 使用方法: node pdf_converter.js <input_html_path> <output_pdf_path>
 *
 * 必要なパッケージ:
 * - puppeteer: `npm install puppeteer`
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

// コマンドライン引数から入力HTMLと出力PDFのパスを取得
const inputHtmlPath = process.argv[2];
const outputPdfPath = process.argv[3];

if (!inputHtmlPath || !outputPdfPath) {
  console.error('使用方法: node pdf_converter.js <input_html_path> <output_pdf_path>');
  process.exit(1);
}

/**
 * HTMLファイルをPDFに変換する
 */
async function convertHtmlToPdf() {
  // HTMLファイルの存在確認
  if (!fs.existsSync(inputHtmlPath)) {
    console.error(`エラー: 入力HTMLファイル "${inputHtmlPath}" が見つかりません`);
    process.exit(1);
  }

  // 出力ディレクトリの存在確認と作成
  const outputDir = path.dirname(outputPdfPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  console.log('HTMLからPDFへの変換を開始します...');
  console.log(`入力: ${inputHtmlPath}`);
  console.log(`出力: ${outputPdfPath}`);

  let browser;

  try {
    // Puppeteerブラウザの起動
    browser = await puppeteer.launch({
      headless: 'new',  // 新しいヘッドレスモード
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
    });

    // 新しいページを作成
    const page = await browser.newPage();

    // HTMLファイルの読み込み
    const htmlContent = fs.readFileSync(inputHtmlPath, 'utf8');
    await page.setContent(htmlContent, {
      waitUntil: 'networkidle0', // すべてのリソースが読み込まれるまで待機
    });

    // フォントとスタイルが適用されるのを待機
    await page.waitForTimeout(1000);

    // PDFオプションの設定
    const pdfOptions = {
      path: outputPdfPath,
      format: 'A4',
      printBackground: true,
      margin: {
        top: '1cm',
        right: '1cm',
        bottom: '1cm',
        left: '1cm',
      },
      displayHeaderFooter: true,
      headerTemplate: `
        <div style="width: 100%; font-size: 8px; text-align: center; color: #777;">
          <span>スタートアップウェルネス分析レポート</span>
        </div>
      `,
      footerTemplate: `
        <div style="width: 100%; font-size: 8px; text-align: center; color: #777;">
          <span>ページ <span class="pageNumber"></span> / <span class="totalPages"></span></span>
        </div>
      `,
      preferCSSPageSize: true,
    };

    // PDF生成
    await page.pdf(pdfOptions);
    console.log('PDF変換が完了しました');
  } catch (error) {
    console.error('PDF変換中にエラーが発生しました:', error);
    process.exit(1);
  } finally {
    // ブラウザを閉じる
    if (browser) {
      await browser.close();
    }
  }
}

// 変換処理を実行
convertHtmlToPdf().catch(error => {
  console.error('予期せぬエラーが発生しました:', error);
  process.exit(1);
});