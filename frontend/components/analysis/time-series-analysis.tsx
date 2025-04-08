"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { TimeSeriesChart } from "@/components/charts/time-series-chart"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const TimeSeriesAnalysis = memo(function TimeSeriesAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">時系列分析</h2>
          <p className="text-sm text-text-secondary">企業のウェルネススコアと成長率の時間的変化を分析します。</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select defaultValue="3m">
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

      <Tabs defaultValue="wellness">
        <TabsList className="mb-4">
          <TabsTrigger value="wellness">ウェルネススコア</TabsTrigger>
          <TabsTrigger value="growth">成長率</TabsTrigger>
          <TabsTrigger value="combined">複合指標</TabsTrigger>
          <TabsTrigger value="comparison">企業比較</TabsTrigger>
        </TabsList>

        <TabsContent value="wellness" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>ウェルネススコア推移</CardTitle>
              <CardDescription>過去3ヶ月間の企業ごとのウェルネススコア推移</CardDescription>
            </CardHeader>
            <CardContent>
              <TimeSeriesChart type="wellness" />
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>統計サマリー</CardTitle>
                <CardDescription>ウェルネススコアの統計的概要</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">平均値</div>
                      <div className="text-2xl font-bold">72.4</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">中央値</div>
                      <div className="text-2xl font-bold">75.1</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">最大値</div>
                      <div className="text-2xl font-bold">92.5</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">最小値</div>
                      <div className="text-2xl font-bold">42.3</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>変動分析</CardTitle>
                <CardDescription>ウェルネススコアの変動と傾向</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">標準偏差</div>
                      <div className="text-2xl font-bold">12.8</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">変動係数</div>
                      <div className="text-2xl font-bold">0.18</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">トレンド</div>
                      <div className="text-2xl font-bold text-secondary">上昇</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">季節性</div>
                      <div className="text-2xl font-bold">弱</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="growth" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>成長率推移</CardTitle>
              <CardDescription>過去3ヶ月間の企業ごとの成長率推移</CardDescription>
            </CardHeader>
            <CardContent>
              <TimeSeriesChart type="growth" />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="combined" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>複合指標分析</CardTitle>
              <CardDescription>ウェルネススコアと成長率の複合指標</CardDescription>
            </CardHeader>
            <CardContent>
              <TimeSeriesChart type="combined" />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="comparison" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>企業比較</CardTitle>
              <CardDescription>選択した企業間のウェルネススコア比較</CardDescription>
            </CardHeader>
            <CardContent>
              <TimeSeriesChart type="comparison" />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})

