"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const MarketAnalysis = memo(function MarketAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">市場・競合分析</h2>
          <p className="text-sm text-text-secondary">市場規模推定、競合マッピング、ポジショニング分析などを行います。</p>
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

      <Tabs defaultValue="market-size">
        <TabsList className="mb-4">
          <TabsTrigger value="market-size">市場規模</TabsTrigger>
          <TabsTrigger value="competitor-map">競合マッピング</TabsTrigger>
          <TabsTrigger value="positioning">ポジショニング</TabsTrigger>
          <TabsTrigger value="trends">市場トレンド</TabsTrigger>
        </TabsList>

        <TabsContent value="market-size" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>市場規模推定</CardTitle>
              <CardDescription>TAM/SAM/SOM市場規模の推定と将来予測</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                市場規模推定のグラフがここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>市場区分</CardTitle>
                <CardDescription>市場規模推定の詳細</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">TAM（総対象市場）</div>
                      <div className="text-2xl font-bold">$12.5B</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">SAM（実行可能市場）</div>
                      <div className="text-2xl font-bold">$3.8B</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">SOM（獲得可能市場）</div>
                      <div className="text-2xl font-bold">$750M</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">年間成長率</div>
                      <div className="text-2xl font-bold">18.3%</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>地域別市場規模</CardTitle>
                <CardDescription>主要地域の市場シェア</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  地域別市場規模のチャートがここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="competitor-map" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>競合マッピング</CardTitle>
              <CardDescription>主要な競合企業のマッピング（PCAによる次元削減）</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                競合マッピングの図がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="positioning" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>ポジショニング分析</CardTitle>
              <CardDescription>競合間のポジショニング比較（レーダーチャート）</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                ポジショニング分析のレーダーチャートがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trends" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>市場トレンド分析</CardTitle>
              <CardDescription>キーワードデータに基づく市場トレンドの分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                市場トレンド分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})