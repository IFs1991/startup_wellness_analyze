"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const TeamAnalysis = memo(function TeamAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">チーム・組織分析</h2>
          <p className="text-sm text-text-secondary">創業チーム評価、組織成長分析、文化・エンゲージメント強度測定などを行います。</p>
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

      <Tabs defaultValue="founder-team">
        <TabsList className="mb-4">
          <TabsTrigger value="founder-team">創業チーム</TabsTrigger>
          <TabsTrigger value="org-growth">組織成長</TabsTrigger>
          <TabsTrigger value="culture">文化・エンゲージメント</TabsTrigger>
          <TabsTrigger value="network">組織ネットワーク</TabsTrigger>
        </TabsList>

        <TabsContent value="founder-team" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>創業チーム評価</CardTitle>
              <CardDescription>創業チームの経験、スキル、多様性の分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                創業チーム評価のグラフがここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>チーム強度</CardTitle>
                <CardDescription>創業チームの強みと弱み</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">経験スコア</div>
                      <div className="text-2xl font-bold">8.4/10</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">スキル補完性</div>
                      <div className="text-2xl font-bold">9.2/10</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">多様性指数</div>
                      <div className="text-2xl font-bold">7.5/10</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">総合評価</div>
                      <div className="text-2xl font-bold">8.3/10</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>メンバー分析</CardTitle>
                <CardDescription>主要メンバーの個別分析</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  メンバー分析のチャートがここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="org-growth" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>組織成長分析</CardTitle>
              <CardDescription>組織の成長パターンと効率性の分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                組織成長分析のグラフがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="culture" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>文化・エンゲージメント強度測定</CardTitle>
              <CardDescription>組織文化とメンバーエンゲージメントの測定</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                文化・エンゲージメント強度の分析結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="network" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>組織ネットワーク分析</CardTitle>
              <CardDescription>組織内の情報・知識の流れの可視化</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                組織ネットワークグラフがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})