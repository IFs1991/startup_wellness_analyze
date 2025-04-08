"use client"

import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ChevronLeft, ChevronRight, TrendingUp, TrendingDown, AlertTriangle } from "lucide-react"
import { cn } from "@/lib/utils"
import { memo } from "react"

interface RightSidebarProps {
  collapsed: boolean
  onToggleCollapse: () => void
}

export const RightSidebar = memo(function RightSidebar({ collapsed, onToggleCollapse }: RightSidebarProps) {
  return (
    <div
      className={cn(
        "relative flex h-full flex-col border-l border-background-lighter bg-background-light transition-all duration-300",
        collapsed ? "w-16" : "w-80",
      )}
    >
      <Button
        variant="ghost"
        size="icon"
        className="absolute -left-3 top-4 z-10 h-6 w-6 rounded-full border border-background-lighter bg-background-light"
        onClick={onToggleCollapse}
      >
        {collapsed ? <ChevronLeft className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
      </Button>

      {!collapsed ? (
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">注目企業</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm">テックスタート社</span>
                  <div className="flex items-center text-secondary">
                    <TrendingUp className="mr-1 h-3 w-3" />
                    <span className="text-xs">+12%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">イノベーション株式会社</span>
                  <div className="flex items-center text-warning">
                    <TrendingDown className="mr-1 h-3 w-3" />
                    <span className="text-xs">-5%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">フューチャーラボ</span>
                  <div className="flex items-center text-secondary">
                    <TrendingUp className="mr-1 h-3 w-3" />
                    <span className="text-xs">+8%</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">リスクアラート</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-start gap-2 rounded-md bg-background-main p-2">
                  <AlertTriangle className="h-4 w-4 text-warning" />
                  <div className="text-xs">
                    <p className="font-medium">テックビジョン社</p>
                    <p className="text-text-muted">ウェルネススコアが30日間で15%低下</p>
                  </div>
                </div>
                <div className="flex items-start gap-2 rounded-md bg-background-main p-2">
                  <AlertTriangle className="h-4 w-4 text-warning" />
                  <div className="text-xs">
                    <p className="font-medium">グロースエンジン</p>
                    <p className="text-text-muted">成長率が前四半期比で20%減少</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </ScrollArea>
      ) : (
        <div className="flex flex-1 flex-col items-center gap-4 py-4">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <TrendingUp className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <AlertTriangle className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  )
})

