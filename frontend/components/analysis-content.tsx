"use client"

import { useState, lazy, Suspense, memo } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  LayoutDashboard,
  TrendingUp,
  Network,
  Activity,
  MessageSquare,
  Percent,
  ScatterChart,
  Download,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { timeFrameOptions } from "@/lib/constants"

// 遅延ロードするコンポーネント
const DashboardContent = lazy(() =>
  import("@/components/dashboard-content").then((mod) => ({ default: mod.DashboardContent })),
)

// ローディングフォールバック
const TabContentLoading = () => (
  <div className="flex h-[200px] items-center justify-center">
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
  </div>
)

export const AnalysisContent = memo(function AnalysisContent() {
  const [timeFrame, setTimeFrame] = useState("3m")

  return (
    <div className="h-full p-4">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-2xl font-bold">分析ダッシュボード</h2>
        <div className="flex items-center gap-2">
          <Select value={timeFrame} onValueChange={setTimeFrame}>
            <SelectTrigger className="w-[120px]">
              <SelectValue placeholder="期間" />
            </SelectTrigger>
            <SelectContent>
              {timeFrameOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon">
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Tabs defaultValue="dashboard">
        <TabsList className="mb-4">
          <TabsTrigger value="dashboard" className="flex items-center gap-1">
            <LayoutDashboard className="h-4 w-4" />
            <span>ダッシュボード</span>
          </TabsTrigger>
          <TabsTrigger value="time-series" className="flex items-center gap-1">
            <TrendingUp className="h-4 w-4" />
            <span>時系列分析</span>
          </TabsTrigger>
          <TabsTrigger value="clustering" className="flex items-center gap-1">
            <ScatterChart className="h-4 w-4" />
            <span>クラスタリング</span>
          </TabsTrigger>
          <TabsTrigger value="correlation" className="flex items-center gap-1">
            <Network className="h-4 w-4" />
            <span>相関分析</span>
          </TabsTrigger>
          <TabsTrigger value="survival" className="flex items-center gap-1">
            <Activity className="h-4 w-4" />
            <span>生存分析</span>
          </TabsTrigger>
          <TabsTrigger value="text-mining" className="flex items-center gap-1">
            <MessageSquare className="h-4 w-4" />
            <span>テキストマイニング</span>
          </TabsTrigger>
          <TabsTrigger value="bayesian" className="flex items-center gap-1">
            <Percent className="h-4 w-4" />
            <span>ベイジアン分析</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="mt-0">
          <Suspense fallback={<TabContentLoading />}>
            <DashboardContent />
          </Suspense>
        </TabsContent>
        <TabsContent value="time-series" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>時系列分析</CardTitle>
              <CardDescription>企業のウェルネススコアと成長率の時間的変化を分析します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                時系列分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="clustering" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>クラスタリング</CardTitle>
              <CardDescription>類似した特性を持つ企業のグループを特定します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                クラスタリング分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="correlation" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>相関分析</CardTitle>
              <CardDescription>異なる指標間の関係性を分析します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                相関分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="survival" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>生存分析</CardTitle>
              <CardDescription>企業の存続確率と関連要因を分析します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                生存分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="text-mining" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>テキストマイニング</CardTitle>
              <CardDescription>企業の説明文やレポートから重要な洞察を抽出します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                テキストマイニングのコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="bayesian" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>ベイジアン分析</CardTitle>
              <CardDescription>確率論的アプローチで企業の将来性を予測します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                ベイジアン分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})

export default { AnalysisContent }

