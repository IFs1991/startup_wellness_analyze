"use client"

import type React from "react"

import { useState } from "react"
import { LeftSidebar } from "@/components/left-sidebar"
import { RightSidebar } from "@/components/right-sidebar"
import { useMobile } from "@/hooks/use-mobile"

export function AnalysisLayout({ children }: { children: React.ReactNode }) {
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
      <div className="flex-1 overflow-auto">{children}</div>

      {/* Right Sidebar - Hidden on mobile */}
      {!isMobile && (
        <RightSidebar
          collapsed={rightSidebarCollapsed}
          onToggleCollapse={() => setRightSidebarCollapsed(!rightSidebarCollapsed)}
        />
      )}
    </div>
  )
}

