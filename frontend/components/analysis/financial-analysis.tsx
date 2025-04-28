"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const FinancialAnalysis = memo(function FinancialAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">財務分析</h2>
          <p className="text-sm text-text-secondary">キャッシュバーン率、ユニットエコノミクス、成長指標などの財務分析を行います。</p>
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

      <Tabs defaultValue="burn-rate">
        <TabsList className="mb-4">
          <TabsTrigger value="burn-rate">キャッシュバーン</TabsTrigger>
          <TabsTrigger value="unit-economics">ユニットエコノミクス</TabsTrigger>
          <TabsTrigger value="growth">成長指標</TabsTrigger>
          <TabsTrigger value="funding">資金調達効率</TabsTrigger>
        </TabsList>

        <TabsContent value="burn-rate" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>キャッシュバーン分析</CardTitle>
              <CardDescription>キャッシュバーン率とランウェイの計算</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                キャッシュバーン分析のグラフがここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>現在のバーン状況</CardTitle>
                <CardDescription>直近のキャッシュバーン指標</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">月間バーン率</div>
                      <div className="text-2xl font-bold">$125K</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">四半期バーン率</div>
                      <div className="text-2xl font-bold">$375K</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">現金残高</div>
                      <div className="text-2xl font-bold">$1.8M</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">ランウェイ</div>
                      <div className="text-2xl font-bold">14.4月</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>バーン効率</CardTitle>
                <CardDescription>バーンに対する成長の効率性</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">バーン倍率</div>
                      <div className="text-2xl font-bold">2.7x</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">売上高/バーン</div>
                      <div className="text-2xl font-bold">1.2x</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">バーントレンド</div>
                      <div className="text-2xl font-bold text-secondary">減少</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">効率スコア</div>
                      <div className="text-2xl font-bold">B+</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="unit-economics" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>ユニットエコノミクス分析</CardTitle>
              <CardDescription>LTV/CAC比率など顧客単位の収益性分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                ユニットエコノミクス分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="growth" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>成長指標分析</CardTitle>
              <CardDescription>MoM、QoQ、YoY成長率などの成長指標</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                成長指標分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="funding" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>資金調達効率分析</CardTitle>
              <CardDescription>資金調達効率と希釈化分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                資金調達効率分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})