"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const VcRoiAnalysis = memo(function VcRoiAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">VC向けROI計算</h2>
          <p className="text-sm text-text-secondary">投資収益率の計算と予測を行い、リスク調整済みROIを分析します。</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select defaultValue="all">
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
            <Filter className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon">
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon">
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Tabs defaultValue="basic-roi">
        <TabsList className="mb-4">
          <TabsTrigger value="basic-roi">基本ROI</TabsTrigger>
          <TabsTrigger value="risk-adjusted">リスク調整済み</TabsTrigger>
          <TabsTrigger value="health-impact">健康影響考慮</TabsTrigger>
          <TabsTrigger value="sensitivity">感度分析</TabsTrigger>
        </TabsList>

        <TabsContent value="basic-roi" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>基本ROI計算</CardTitle>
              <CardDescription>投資収益率の基本計算と時系列推移</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                基本ROI計算のグラフがここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>ROIサマリー</CardTitle>
                <CardDescription>現在のROI統計</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">シンプルROI</div>
                      <div className="text-2xl font-bold">3.4x</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">IRR</div>
                      <div className="text-2xl font-bold">24.8%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">回収期間</div>
                      <div className="text-2xl font-bold">4.2年</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">TVPI</div>
                      <div className="text-2xl font-bold">2.8x</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>ファンド比較</CardTitle>
                <CardDescription>類似ファンドとのROI比較</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  ファンド比較のチャートがここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="risk-adjusted" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>リスク調整済みROI</CardTitle>
              <CardDescription>リスクを考慮した投資収益率分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                リスク調整済みROI分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="health-impact" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>健康影響考慮ROI</CardTitle>
              <CardDescription>ウェルネス要因を考慮した投資収益率</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                健康影響考慮ROI分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sensitivity" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>ROI感度分析</CardTitle>
              <CardDescription>各要因のROIへの影響度分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                ROI感度分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})