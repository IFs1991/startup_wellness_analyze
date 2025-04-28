"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const PortfolioAnalysis = memo(function PortfolioAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">ポートフォリオネットワーク分析</h2>
          <p className="text-sm text-text-secondary">企業間のネットワーク関係を構築・分析し、エコシステム係数を計算します。</p>
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

      <Tabs defaultValue="network-graph">
        <TabsList className="mb-4">
          <TabsTrigger value="network-graph">ネットワークグラフ</TabsTrigger>
          <TabsTrigger value="centrality">中心性分析</TabsTrigger>
          <TabsTrigger value="communities">コミュニティ検出</TabsTrigger>
          <TabsTrigger value="ecosystem">エコシステム係数</TabsTrigger>
        </TabsList>

        <TabsContent value="network-graph" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>企業間ネットワークグラフ</CardTitle>
              <CardDescription>企業間の関係性とつながりの可視化</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                ネットワークグラフがここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>ネットワーク統計</CardTitle>
                <CardDescription>ネットワーク全体の特性</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">ネットワーク密度</div>
                      <div className="text-2xl font-bold">0.24</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">平均経路長</div>
                      <div className="text-2xl font-bold">2.7</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">クラスタ係数</div>
                      <div className="text-2xl font-bold">0.68</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">連結成分数</div>
                      <div className="text-2xl font-bold">3</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>主要企業</CardTitle>
                <CardDescription>ネットワーク内で影響力の高い企業</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  主要企業のリストがここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="centrality" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>中心性分析</CardTitle>
              <CardDescription>企業の中心性と影響力の測定</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                中心性分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="communities" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>コミュニティ検出</CardTitle>
              <CardDescription>密接につながった企業グループの特定</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                コミュニティ検出の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ecosystem" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>エコシステム係数分析</CardTitle>
              <CardDescription>ポートフォリオ全体のエコシステム効果</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                エコシステム係数分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})