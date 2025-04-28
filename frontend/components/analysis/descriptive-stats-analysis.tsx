"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const DescriptiveStatsAnalysis = memo(function DescriptiveStatsAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">記述統計</h2>
          <p className="text-sm text-text-secondary">平均、中央値、標準偏差などの基本的な統計量を計算します。</p>
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

      <Tabs defaultValue="summary">
        <TabsList className="mb-4">
          <TabsTrigger value="summary">基本統計量</TabsTrigger>
          <TabsTrigger value="distribution">分布分析</TabsTrigger>
          <TabsTrigger value="outliers">外れ値分析</TabsTrigger>
          <TabsTrigger value="comparison">企業間比較</TabsTrigger>
        </TabsList>

        <TabsContent value="summary" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>基本統計量</CardTitle>
              <CardDescription>各指標の基本的な統計情報</CardDescription>
            </CardHeader>
            <CardContent className="h-[350px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                基本統計量の結果がここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>中心傾向</CardTitle>
                <CardDescription>データの中心的な値を示す統計量</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">平均値</div>
                      <div className="text-2xl font-bold">68.4</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">中央値</div>
                      <div className="text-2xl font-bold">72.1</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">最頻値</div>
                      <div className="text-2xl font-bold">75.0</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">調和平均</div>
                      <div className="text-2xl font-bold">65.2</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>ばらつき</CardTitle>
                <CardDescription>データのばらつきを示す統計量</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">標準偏差</div>
                      <div className="text-2xl font-bold">12.7</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">分散</div>
                      <div className="text-2xl font-bold">161.29</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">範囲</div>
                      <div className="text-2xl font-bold">45.8</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">四分位範囲</div>
                      <div className="text-2xl font-bold">18.6</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="distribution" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>分布分析</CardTitle>
              <CardDescription>データの分布特性の詳細分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                分布分析のグラフと結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="outliers" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>外れ値分析</CardTitle>
              <CardDescription>異常値や外れ値の検出と分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                外れ値分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="comparison" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>企業間比較</CardTitle>
              <CardDescription>選択した企業間の統計量比較</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                企業間比較の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})