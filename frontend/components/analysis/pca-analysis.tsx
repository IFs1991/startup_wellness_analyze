"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const PcaAnalysis = memo(function PcaAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">主成分分析</h2>
          <p className="text-sm text-text-secondary">データの次元削減を行い、重要な特徴を抽出します。</p>
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

      <Tabs defaultValue="visualization">
        <TabsList className="mb-4">
          <TabsTrigger value="visualization">可視化</TabsTrigger>
          <TabsTrigger value="loadings">成分負荷量</TabsTrigger>
          <TabsTrigger value="scree">スクリープロット</TabsTrigger>
          <TabsTrigger value="biplot">バイプロット</TabsTrigger>
        </TabsList>

        <TabsContent value="visualization" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>主成分分析の可視化</CardTitle>
              <CardDescription>2次元平面上での企業分布</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                主成分分析の可視化がここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>寄与率</CardTitle>
                <CardDescription>各主成分の説明力</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">第1主成分</div>
                      <div className="text-2xl font-bold">42.7%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">第2主成分</div>
                      <div className="text-2xl font-bold">27.4%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">第3主成分</div>
                      <div className="text-2xl font-bold">14.8%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">累積寄与率</div>
                      <div className="text-2xl font-bold">84.9%</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>主成分の解釈</CardTitle>
                <CardDescription>主要な主成分の意味</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">第1主成分</div>
                      <div className="text-lg font-medium">企業の安定性・成熟度</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">第2主成分</div>
                      <div className="text-lg font-medium">成長ポテンシャル・革新性</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="loadings" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>成分負荷量</CardTitle>
              <CardDescription>各変数の主成分への寄与度</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                成分負荷量の表がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="scree" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>スクリープロット</CardTitle>
              <CardDescription>主成分の固有値のプロット</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                スクリープロットがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="biplot" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>バイプロット</CardTitle>
              <CardDescription>主成分空間における変数と観測値の関係</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                バイプロットがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})