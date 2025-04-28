"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const CorrelationAnalysis = memo(function CorrelationAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">相関分析</h2>
          <p className="text-sm text-text-secondary">異なる指標間の関係性を分析します。</p>
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

      <Tabs defaultValue="matrix">
        <TabsList className="mb-4">
          <TabsTrigger value="matrix">相関行列</TabsTrigger>
          <TabsTrigger value="scatter">散布図</TabsTrigger>
          <TabsTrigger value="partial">偏相関</TabsTrigger>
          <TabsTrigger value="time-lag">時間ラグ相関</TabsTrigger>
        </TabsList>

        <TabsContent value="matrix" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>相関行列ヒートマップ</CardTitle>
              <CardDescription>変数間の相関係数を視覚化</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                相関ヒートマップがここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>主要相関ペア</CardTitle>
                <CardDescription>強い相関がある変数ペア</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">ウェルネススコア ↔ 従業員定着率</div>
                      <div className="text-2xl font-bold">r = 0.84</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">健康投資額 ↔ 生産性指標</div>
                      <div className="text-2xl font-bold">r = 0.72</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">エコシステム係数 ↔ 成長率</div>
                      <div className="text-2xl font-bold">r = 0.68</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>相関解釈</CardTitle>
                <CardDescription>相関分析の洞察</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  相関分析の洞察がここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="scatter" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>散布図マトリックス</CardTitle>
              <CardDescription>変数間の関係を散布図で表示</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                散布図マトリックスがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="partial" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>偏相関分析</CardTitle>
              <CardDescription>他の変数の影響を制御した相関関係</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                偏相関分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="time-lag" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>時間ラグ相関分析</CardTitle>
              <CardDescription>時間差を考慮した変数間の関係</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                時間ラグ相関分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})