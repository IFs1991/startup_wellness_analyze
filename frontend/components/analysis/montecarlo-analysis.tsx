"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const MontecarloAnalysis = memo(function MontecarloAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">モンテカルロシミュレーション</h2>
          <p className="text-sm text-text-secondary">様々なシナリオに基づいたシミュレーションを実行し、将来予測を行います。</p>
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

      <Tabs defaultValue="cash-flow">
        <TabsList className="mb-4">
          <TabsTrigger value="cash-flow">キャッシュフロー</TabsTrigger>
          <TabsTrigger value="runway">ランウェイ</TabsTrigger>
          <TabsTrigger value="growth">成長シミュレーション</TabsTrigger>
          <TabsTrigger value="scenarios">シナリオ分析</TabsTrigger>
        </TabsList>

        <TabsContent value="cash-flow" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>キャッシュフローシミュレーション</CardTitle>
              <CardDescription>モンテカルロ法による将来キャッシュフローの予測</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                キャッシュフローシミュレーションのグラフがここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>統計サマリー</CardTitle>
                <CardDescription>シミュレーション結果の統計</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">期待値</div>
                      <div className="text-2xl font-bold">$2.4M</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">標準偏差</div>
                      <div className="text-2xl font-bold">$750K</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">5%分位</div>
                      <div className="text-2xl font-bold">$1.2M</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">95%分位</div>
                      <div className="text-2xl font-bold">$3.8M</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>主要リスク要因</CardTitle>
                <CardDescription>結果に最も影響を与える要因</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  主要リスク要因のチャートがここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="runway" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>ランウェイシミュレーション</CardTitle>
              <CardDescription>資金枯渇までの期間の確率分布</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                ランウェイシミュレーションの結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="growth" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>成長シミュレーション</CardTitle>
              <CardDescription>売上・ユーザー数などの成長予測</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                成長シミュレーションの結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="scenarios" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>シナリオ分析</CardTitle>
              <CardDescription>複数の将来シナリオの比較分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                シナリオ分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})