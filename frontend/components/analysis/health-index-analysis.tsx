"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const HealthIndexAnalysis = memo(function HealthIndexAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">健康投資効果指数計算</h2>
          <p className="text-sm text-text-secondary">健康投資効果指数（HIEI）を計算し、エコシステム影響度を分析します。</p>
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

      <Tabs defaultValue="hiei">
        <TabsList className="mb-4">
          <TabsTrigger value="hiei">HIEI基本計算</TabsTrigger>
          <TabsTrigger value="ecosystem">エコシステム影響</TabsTrigger>
          <TabsTrigger value="benchmark">業界ベンチマーク</TabsTrigger>
          <TabsTrigger value="team">チーム構成考慮</TabsTrigger>
        </TabsList>

        <TabsContent value="hiei" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>健康投資効果指数</CardTitle>
              <CardDescription>健康投資が企業パフォーマンスに与える影響の定量化</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                健康投資効果指数のグラフがここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>HIEIコンポーネント</CardTitle>
                <CardDescription>指数を構成する要素</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">健康投資額</div>
                      <div className="text-2xl font-bold">$245K</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">健康ROI</div>
                      <div className="text-2xl font-bold">3.2x</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">生産性向上率</div>
                      <div className="text-2xl font-bold">12.7%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">HIEI総合</div>
                      <div className="text-2xl font-bold">84.3/100</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>時系列推移</CardTitle>
                <CardDescription>HIEIの時間的変化</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  HIEIの時系列チャートがここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="ecosystem" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>エコシステム影響度</CardTitle>
              <CardDescription>健康投資のエコシステム全体への波及効果</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                エコシステム影響度の分析結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="benchmark" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>業界ベンチマーク</CardTitle>
              <CardDescription>業界平均と比較したHIEIの位置付け</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                業界ベンチマーク分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="team" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>チーム構成考慮HIEI</CardTitle>
              <CardDescription>役職・チーム構成を考慮したHIEIの調整計算</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                チーム構成考慮HIEI計算の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})