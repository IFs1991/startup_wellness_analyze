"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const AssociationAnalysis = memo(function AssociationAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">アソシエーション分析</h2>
          <p className="text-sm text-text-secondary">アイテム間の関連ルールを見つけ出します（例：Aprioriアルゴリズム）。</p>
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

      <Tabs defaultValue="rules">
        <TabsList className="mb-4">
          <TabsTrigger value="rules">関連ルール</TabsTrigger>
          <TabsTrigger value="patterns">パターン分析</TabsTrigger>
          <TabsTrigger value="matrix">マトリックス表示</TabsTrigger>
          <TabsTrigger value="network">ネットワーク表示</TabsTrigger>
        </TabsList>

        <TabsContent value="rules" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>関連ルール一覧</CardTitle>
              <CardDescription>サポート値、信頼度、リフト値によるフィルタリングが可能です</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                アソシエーション分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>統計サマリー</CardTitle>
                <CardDescription>発見された関連ルールの統計的概要</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">ルール総数</div>
                      <div className="text-2xl font-bold">124</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">平均信頼度</div>
                      <div className="text-2xl font-bold">0.72</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">平均サポート</div>
                      <div className="text-2xl font-bold">0.15</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">平均リフト</div>
                      <div className="text-2xl font-bold">2.4</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>トップルール</CardTitle>
                <CardDescription>信頼度の高い上位ルール</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  トップルールの表示エリア
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="patterns" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>パターン分析</CardTitle>
              <CardDescription>頻出アイテムセットと時系列パターン</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                パターン分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="matrix" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>マトリックス表示</CardTitle>
              <CardDescription>アイテム間の関連度をマトリックス形式で表示</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                マトリックス表示がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="network" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>ネットワーク表示</CardTitle>
              <CardDescription>アイテム間の関連をネットワークグラフで表示</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                ネットワーク表示がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})