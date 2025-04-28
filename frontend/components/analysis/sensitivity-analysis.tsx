"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const SensitivityAnalysis = memo(function SensitivityAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">感度分析</h2>
          <p className="text-sm text-text-secondary">パラメータの変化が出力に与える影響を分析し、トルネードチャートを生成します。</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select defaultValue="12m">
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

      <Tabs defaultValue="tornado">
        <TabsList className="mb-4">
          <TabsTrigger value="tornado">トルネードチャート</TabsTrigger>
          <TabsTrigger value="one-way">1方向感度分析</TabsTrigger>
          <TabsTrigger value="two-way">2方向感度分析</TabsTrigger>
          <TabsTrigger value="critical">臨界値分析</TabsTrigger>
        </TabsList>

        <TabsContent value="tornado" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>トルネードチャート分析</CardTitle>
              <CardDescription>各パラメータの結果に対する影響度の可視化</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                トルネードチャートがここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>主要パラメータ</CardTitle>
                <CardDescription>最も影響の大きいパラメータ</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">顧客獲得コスト</div>
                      <div className="text-2xl font-bold">42.3%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">顧客生涯価値</div>
                      <div className="text-2xl font-bold">38.7%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">解約率</div>
                      <div className="text-2xl font-bold">29.4%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">運用コスト</div>
                      <div className="text-2xl font-bold">21.8%</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>感度係数</CardTitle>
                <CardDescription>出力の弾力性</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  感度係数のチャートがここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="one-way" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>1方向感度分析</CardTitle>
              <CardDescription>単一パラメータの変化による影響分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                1方向感度分析のグラフがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="two-way" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>2方向感度分析</CardTitle>
              <CardDescription>2つのパラメータの同時変化による影響分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                2方向感度分析のヒートマップがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="critical" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>臨界値分析</CardTitle>
              <CardDescription>結果が急激に変化する臨界値の特定</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                臨界値分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})