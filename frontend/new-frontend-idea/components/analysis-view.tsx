"use client"

import { useState, lazy, Suspense, memo } from "react"
import { LeftSidebar } from "@/components/left-sidebar"
import { RightSidebar } from "@/components/right-sidebar"
import { useMobile } from "@/hooks/use-mobile"

// 遅延ロードするコンポーネント
const AnalysisContent = lazy(() =>
  import("@/components/analysis-content").then((mod) => ({ default: mod.AnalysisContent })),
)

// ローディングフォールバック
const ContentLoading = () => (
  <div className="flex h-full items-center justify-center">
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
  </div>
)

export const AnalysisView = memo(function AnalysisView() {
  const [leftSidebarCollapsed, setLeftSidebarCollapsed] = useState(false)
  const [rightSidebarCollapsed, setRightSidebarCollapsed] = useState(false)
  const isMobile = useMobile()

  return (
    <div className="flex h-full">
      {/* Left Sidebar - Hidden on mobile */}
      {!isMobile && (
        <LeftSidebar
          collapsed={leftSidebarCollapsed}
          onToggleCollapse={() => setLeftSidebarCollapsed(!leftSidebarCollapsed)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        <Suspense fallback={<ContentLoading />}>
          <AnalysisContent />
        </Suspense>
      </div>

      {/* Right Sidebar - Hidden on mobile */}
      {!isMobile && (
        <RightSidebar
          collapsed={rightSidebarCollapsed}
          onToggleCollapse={() => setRightSidebarCollapsed(!rightSidebarCollapsed)}
        />
      )}
    </div>
  )
})

export default { AnalysisView }

