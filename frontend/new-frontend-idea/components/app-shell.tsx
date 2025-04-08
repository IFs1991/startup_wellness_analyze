"use client"

import { useState, lazy, Suspense } from "react"
import { ViewSwitcher } from "@/components/view-switcher"
import { Header } from "@/components/header"
import { UserMenu } from "@/components/user-menu"
import { Logo } from "@/components/logo"

// 遅延ロードするコンポーネント
const ChatView = lazy(() => import("@/components/chat-view").then((mod) => ({ default: mod.ChatView })))
const AnalysisView = lazy(() => import("@/components/analysis-view").then((mod) => ({ default: mod.AnalysisView })))

// ローディングフォールバック
const ViewLoading = () => (
  <div className="flex h-full items-center justify-center">
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
  </div>
)

type ViewType = "chat" | "analysis"

export function AppShell() {
  const [currentView, setCurrentView] = useState<ViewType>("chat")

  return (
    <div className="flex h-screen flex-col bg-background-main text-text-primary">
      <Header>
        <div className="flex items-center gap-4">
          <Logo />
          <h1 className="text-xl font-bold">スタートアップウェルネスアナライザー</h1>
        </div>
        <div className="flex items-center gap-4">
          <ViewSwitcher currentView={currentView} onViewChange={setCurrentView} />
          <UserMenu />
        </div>
      </Header>
      <main className="flex-1 overflow-hidden">
        <Suspense fallback={<ViewLoading />}>{currentView === "chat" ? <ChatView /> : <AnalysisView />}</Suspense>
      </main>
    </div>
  )
}

