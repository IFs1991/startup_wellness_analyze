"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const SurvivalAnalysis = memo(function SurvivalAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">生存分析</h2>
          <p className="text-sm text-text-secondary">企業の存続確率と関連要因を分析します。</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select defaultValue="60m">
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

      <Tabs defaultValue="kaplan-meier">
        <TabsList className="mb-4">
          <TabsTrigger value="kaplan-meier">カプランマイヤー曲線</TabsTrigger>
          <TabsTrigger value="cox">コックス比例ハザード</TabsTrigger>
          <TabsTrigger value="hazard">ハザード関数</TabsTrigger>
          <TabsTrigger value="competing">競合リスク</TabsTrigger>
        </TabsList>

        <TabsContent value="kaplan-meier" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>カプランマイヤー生存曲線</CardTitle>
              <CardDescription>時間経過に伴う企業の生存確率</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                カプランマイヤー曲線のグラフがここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>生存統計</CardTitle>
                <CardDescription>主要な生存指標</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">5年生存率</div>
                      <div className="text-2xl font-bold">76.3%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">中央生存期間</div>
                      <div className="text-2xl font-bold">8.4年</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">10年生存率</div>
                      <div className="text-2xl font-bold">42.7%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">信頼区間</div>
                      <div className="text-2xl font-bold">±5.2%</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>集団比較</CardTitle>
                <CardDescription>異なる企業グループ間の生存率比較</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  集団比較のグラフがここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="cox" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>コックス比例ハザード分析</CardTitle>
              <CardDescription>複数の要因が生存に与える影響の評価</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                コックス比例ハザードモデルの結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="hazard" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>ハザード関数分析</CardTitle>
              <CardDescription>時間経過に伴うイベント発生リスクの変化</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                ハザード関数のグラフがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="competing" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>競合リスク分析</CardTitle>
              <CardDescription>複数の異なるイベントタイプを考慮した生存分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                競合リスク分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})