"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const BayesianAnalysis = memo(function BayesianAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">ベイジアン分析</h2>
          <p className="text-sm text-text-secondary">確率論的アプローチで企業の将来性を予測します。</p>
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

      <Tabs defaultValue="posterior">
        <TabsList className="mb-4">
          <TabsTrigger value="posterior">事後確率分布</TabsTrigger>
          <TabsTrigger value="prior">事前確率設定</TabsTrigger>
          <TabsTrigger value="prediction">予測分布</TabsTrigger>
          <TabsTrigger value="mcmc">MCMCシミュレーション</TabsTrigger>
        </TabsList>

        <TabsContent value="posterior" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>事後確率分布</CardTitle>
              <CardDescription>データに基づいた更新後の確率分布</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                事後確率分布のグラフがここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>パラメータ推定</CardTitle>
                <CardDescription>ベイズ推定によるパラメータ値</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">成長率μ</div>
                      <div className="text-2xl font-bold">8.4%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">分散σ²</div>
                      <div className="text-2xl font-bold">0.023</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">信頼区間</div>
                      <div className="text-2xl font-bold">±2.7%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">ベイズファクター</div>
                      <div className="text-2xl font-bold">12.3</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>モデル比較</CardTitle>
                <CardDescription>異なるモデルの比較と評価</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  モデル比較の結果がここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="prior" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>事前確率設定</CardTitle>
              <CardDescription>モデルパラメータの事前分布の設定</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                事前確率設定のフォームがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="prediction" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>予測分布</CardTitle>
              <CardDescription>将来のデータに対する予測確率分布</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                予測分布のグラフがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="mcmc" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>MCMCシミュレーション</CardTitle>
              <CardDescription>マルコフ連鎖モンテカルロ法による複雑な事後分布の推定</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                MCMCシミュレーションの結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})