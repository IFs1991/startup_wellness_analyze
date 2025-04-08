"use client"

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { MessageSquare, BarChart } from "lucide-react"
import { memo } from "react"

type ViewType = "chat" | "analysis"

interface ViewSwitcherProps {
  currentView: ViewType
  onViewChange: (view: ViewType) => void
}

export const ViewSwitcher = memo(function ViewSwitcher({ currentView, onViewChange }: ViewSwitcherProps) {
  return (
    <Tabs value={currentView} onValueChange={(value) => onViewChange(value as ViewType)} className="w-[200px]">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="chat" className="flex items-center gap-1">
          <MessageSquare className="h-4 w-4" />
          <span>チャット</span>
        </TabsTrigger>
        <TabsTrigger value="analysis" className="flex items-center gap-1">
          <BarChart className="h-4 w-4" />
          <span>分析</span>
        </TabsTrigger>
      </TabsList>
    </Tabs>
  )
})

