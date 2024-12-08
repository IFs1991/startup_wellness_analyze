import * as React from "react"
import { ThemeProvider } from "next-themes"
import { SiteHeader } from "@/components/layout/site-header"
import { UserNav } from "@/components/layout/user-nav"
import { MainNav } from "@/components/layout/main-nav"

interface RootLayoutProps {
  children: React.ReactNode
}

// アプリケーション全体のレイアウトを定義するコンポーネント
export default function RootLayout({ children }: RootLayoutProps) {
  return (
    // htmlタグを追加し、言語設定とhydration警告の抑制を設定
    <html lang="ja" suppressHydrationWarning>
      <body>
        {/* ThemeProviderでアプリケーション全体をラップし、テーマ機能を提供 */}
        <ThemeProvider
          attribute="class"  // クラスベースでテーマを切り替え
          defaultTheme="system"  // デフォルトでシステムのテーマ設定を使用
          enableSystem  // システムテーマの使用を有効化
          disableTransitionOnChange  // テーマ切り替え時の遷移を無効化
        >
          <div className="flex min-h-screen flex-col">
            {/* サイトヘッダーコンポーネント */}
            <SiteHeader />
            {/* メインコンテンツエリア */}
            <main className="flex-1">
              {children}
            </main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}