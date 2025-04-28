"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"

export const TextMiningAnalysis = memo(function TextMiningAnalysis() {
  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">テキストマイニング</h2>
          <p className="text-sm text-text-secondary">企業の説明文やレポートから重要な洞察を抽出します。</p>
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

      <Tabs defaultValue="keyword">
        <TabsList className="mb-4">
          <TabsTrigger value="keyword">キーワード分析</TabsTrigger>
          <TabsTrigger value="sentiment">感情分析</TabsTrigger>
          <TabsTrigger value="topic">トピックモデリング</TabsTrigger>
          <TabsTrigger value="network">言語ネットワーク</TabsTrigger>
        </TabsList>

        <TabsContent value="keyword" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>キーワード分析</CardTitle>
              <CardDescription>テキストから重要なキーワードや概念を抽出</CardDescription>
            </CardHeader>
            <CardContent className="h-[400px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                キーワード分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>頻出語句</CardTitle>
                <CardDescription>高頻度で出現する重要語句</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">ウェルネス</div>
                      <div className="text-2xl font-bold">87回</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">デジタル化</div>
                      <div className="text-2xl font-bold">64回</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">持続可能性</div>
                      <div className="text-2xl font-bold">52回</div>
                    </div>
                    <div className="rounded-lg bg-background-main p-3">
                      <div className="text-sm text-text-secondary">投資効率</div>
                      <div className="text-2xl font-bold">41回</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>ワードクラウド</CardTitle>
                <CardDescription>重要語句の視覚的表現</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                  ワードクラウドがここに表示されます
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="sentiment" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>感情分析</CardTitle>
              <CardDescription>テキストの感情傾向と極性の分析</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                感情分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="topic" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>トピックモデリング</CardTitle>
              <CardDescription>潜在的なトピックの識別とクラスタリング</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                トピックモデリングの結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="network" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>言語ネットワーク分析</CardTitle>
              <CardDescription>単語間の関連性とネットワーク構造の可視化</CardDescription>
            </CardHeader>
            <CardContent className="h-[450px]">
              <div className="rounded-md bg-background-main p-4 text-center text-text-muted">
                言語ネットワーク分析の結果がここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})