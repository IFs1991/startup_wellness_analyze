"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const CausalInferenceAnalysis = memo(function CausalInferenceAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">因果推論</h2>
          <p className="text-sm text-text-secondary">差分の差分法、合成コントロール法などを用いた因果関係の分析を行います。</p>
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

      <Tabs defaultValue="did">
        <TabsList className="mb-4">
          <TabsTrigger value="did">差分の差分法</TabsTrigger>
          <TabsTrigger value="synthetic">合成コントロール法</TabsTrigger>
          <TabsTrigger value="causal-impact">因果インパクト</TabsTrigger>
          <TabsTrigger value="heterogeneous">異質処理効果</TabsTrigger>
        </TabsList>

        <TabsContent value="did" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>差分の差分分析</CardTitle>
              <CardDescription>施策の前後と対象/非対象グループの比較による効果測定</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                差分の差分分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>効果サイズ</CardTitle>
                <CardDescription>施策の推定効果と信頼区間</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">平均処理効果</div>
                      <div className="text-2xl font-bold">+12.7%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">p値</div>
                      <div className="text-2xl font-bold">0.023</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">信頼区間 (下限)</div>
                      <div className="text-2xl font-bold">+5.3%</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">信頼区間 (上限)</div>
                      <div className="text-2xl font-bold">+18.2%</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>並行トレンド検証</CardTitle>
                <CardDescription>前提条件の検証結果</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  並行トレンド検証のグラフがここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="synthetic" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>合成コントロール分析</CardTitle>
              <CardDescription>処理群に類似した合成コントロール群との比較</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                合成コントロール分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="causal-impact" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>因果インパクト分析</CardTitle>
              <CardDescription>介入の時系列的な因果効果の推定</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                因果インパクト分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="heterogeneous" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>異質処理効果分析</CardTitle>
              <CardDescription>サブグループごとの効果の違いを分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                異質処理効果分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})