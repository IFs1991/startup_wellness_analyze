"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const ClusteringAnalysis = memo(function ClusteringAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">クラスタリング</h2>
          <p className="text-sm text-text-secondary">類似した特性を持つ企業のグループを特定します。</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select defaultValue="all">
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

      <Tabs defaultValue="kmeans">
        <TabsList className="mb-4">
          <TabsTrigger value="kmeans">K-means</TabsTrigger>
          <TabsTrigger value="hierarchical">階層的クラスタリング</TabsTrigger>
          <TabsTrigger value="dbscan">DBSCAN</TabsTrigger>
          <TabsTrigger value="evaluation">評価指標</TabsTrigger>
        </TabsList>

        <TabsContent value="kmeans" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>K-meansクラスタリング</CardTitle>
              <CardDescription>データポイントをK個のクラスタに分類</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                K-meansクラスタリングの結果がここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>クラスタ概要</CardTitle>
                <CardDescription>各クラスタの主要特性</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">クラスタ1: 高成長・高ウェルネス</div>
                      <div className="text-2xl font-bold">32社</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">クラスタ2: 安定・中ウェルネス</div>
                      <div className="text-2xl font-bold">48社</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">クラスタ3: 新興・発展途上</div>
                      <div className="text-2xl font-bold">27社</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>クラスタ間比較</CardTitle>
                <CardDescription>クラスタ間の主要指標比較</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  クラスタ間比較のチャートがここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="hierarchical" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>階層的クラスタリング</CardTitle>
              <CardDescription>データポイント間の階層的関係を分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                階層的クラスタリングのデンドログラムがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="dbscan" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>DBSCAN分析</CardTitle>
              <CardDescription>密度ベースのクラスタリング</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                DBSCAN分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="evaluation" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>クラスタリング評価</CardTitle>
              <CardDescription>各クラスタリング手法の評価と比較</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                クラスタリング評価指標の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})