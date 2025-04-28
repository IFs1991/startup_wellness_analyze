"use client"

import { useState, lazy, Suspense, memo } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  LayoutDashboard,
  TrendingUp,
  Network,
  Activity,
  MessageSquare,
  Percent,
  ScatterChart,
  Download,
  GitCompare,
  PieChart,
  BrainCircuit,
  DollarSign,
  LineChart,
  Users,
  Briefcase,
  Calculator,
  BarChart4,
  Workflow,
  Share2,
  Medal,
  Heart,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { timeFrameOptions } from "@/lib/constants"

// 遅延ロードするコンポーネント
const DashboardContent = lazy(() =>
  import("@/components/dashboard-content").then((mod) => ({ default: mod.DashboardContent })),
)

// ローディングフォールバック
const TabContentLoading = () => (
  <div className="flex h-[200px] items-center justify-center">
    <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
  </div>
)

export const AnalysisContent = memo(function AnalysisContent() {
  const [timeFrame, setTimeFrame] = useState("3m")

  return (
    <div className="h-full p-4">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-2xl font-bold">分析ダッシュボード</h2>
        <div className="flex items-center gap-2">
          <Select value={timeFrame} onValueChange={setTimeFrame}>
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
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Tabs defaultValue="dashboard">
        <TabsList className="mb-4">
          <TabsTrigger value="dashboard" className="flex items-center gap-1">
            <LayoutDashboard className="h-4 w-4" />
            <span>ダッシュボード</span>
          </TabsTrigger>
          <TabsTrigger value="time-series" className="flex items-center gap-1">
            <TrendingUp className="h-4 w-4" />
            <span>時系列分析</span>
          </TabsTrigger>
          <TabsTrigger value="clustering" className="flex items-center gap-1">
            <ScatterChart className="h-4 w-4" />
            <span>クラスタリング</span>
          </TabsTrigger>
          <TabsTrigger value="correlation" className="flex items-center gap-1">
            <Network className="h-4 w-4" />
            <span>相関分析</span>
          </TabsTrigger>
          <TabsTrigger value="survival" className="flex items-center gap-1">
            <Activity className="h-4 w-4" />
            <span>生存分析</span>
          </TabsTrigger>
          <TabsTrigger value="text-mining" className="flex items-center gap-1">
            <MessageSquare className="h-4 w-4" />
            <span>テキストマイニング</span>
          </TabsTrigger>
          <TabsTrigger value="bayesian" className="flex items-center gap-1">
            <Percent className="h-4 w-4" />
            <span>ベイジアン分析</span>
          </TabsTrigger>
          <TabsTrigger value="association" className="flex items-center gap-1">
            <GitCompare className="h-4 w-4" />
            <span>アソシエーション分析</span>
          </TabsTrigger>
          <TabsTrigger value="descriptive-stats" className="flex items-center gap-1">
            <PieChart className="h-4 w-4" />
            <span>記述統計</span>
          </TabsTrigger>
          <TabsTrigger value="causal-inference" className="flex items-center gap-1">
            <BrainCircuit className="h-4 w-4" />
            <span>因果推論</span>
          </TabsTrigger>
          <TabsTrigger value="pca" className="flex items-center gap-1">
            <LineChart className="h-4 w-4" />
            <span>主成分分析</span>
          </TabsTrigger>
          <TabsTrigger value="financial" className="flex items-center gap-1">
            <DollarSign className="h-4 w-4" />
            <span>財務分析</span>
          </TabsTrigger>
          <TabsTrigger value="market" className="flex items-center gap-1">
            <BarChart4 className="h-4 w-4" />
            <span>市場・競合分析</span>
          </TabsTrigger>
          <TabsTrigger value="team" className="flex items-center gap-1">
            <Users className="h-4 w-4" />
            <span>チーム・組織分析</span>
          </TabsTrigger>
          <TabsTrigger value="montecarlo" className="flex items-center gap-1">
            <Calculator className="h-4 w-4" />
            <span>モンテカルロシミュレーション</span>
          </TabsTrigger>
          <TabsTrigger value="sensitivity" className="flex items-center gap-1">
            <Workflow className="h-4 w-4" />
            <span>感度分析</span>
          </TabsTrigger>
          <TabsTrigger value="portfolio" className="flex items-center gap-1">
            <Share2 className="h-4 w-4" />
            <span>ポートフォリオネットワーク</span>
          </TabsTrigger>
          <TabsTrigger value="vc-roi" className="flex items-center gap-1">
            <Medal className="h-4 w-4" />
            <span>VC向けROI計算</span>
          </TabsTrigger>
          <TabsTrigger value="health-index" className="flex items-center gap-1">
            <Heart className="h-4 w-4" />
            <span>健康投資効果指数</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="mt-0">
          <Suspense fallback={<TabContentLoading />}>
            <DashboardContent />
          </Suspense>
        </TabsContent>
        <TabsContent value="time-series" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>時系列分析</CardTitle>
              <CardDescription>企業のウェルネススコアと成長率の時間的変化を分析します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                時系列分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="clustering" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>クラスタリング</CardTitle>
              <CardDescription>類似した特性を持つ企業のグループを特定します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                クラスタリング分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="correlation" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>相関分析</CardTitle>
              <CardDescription>異なる指標間の関係性を分析します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                相関分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="survival" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>生存分析</CardTitle>
              <CardDescription>企業の存続確率と関連要因を分析します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                生存分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="text-mining" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>テキストマイニング</CardTitle>
              <CardDescription>企業の説明文やレポートから重要な洞察を抽出します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                テキストマイニングのコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="bayesian" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>ベイジアン分析</CardTitle>
              <CardDescription>確率論的アプローチで企業の将来性を予測します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                ベイジアン分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="association" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>アソシエーション分析</CardTitle>
              <CardDescription>アイテム間の関連ルールを見つけ出します（例：Aprioriアルゴリズム）。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                アソシエーション分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="descriptive-stats" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>記述統計</CardTitle>
              <CardDescription>平均、中央値、標準偏差などの基本的な統計量を計算します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                記述統計のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="causal-inference" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>因果推論</CardTitle>
              <CardDescription>差分の差分法、合成コントロール法などを用いた因果関係の分析を行います。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                因果推論のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="pca" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>主成分分析</CardTitle>
              <CardDescription>データの次元削減を行い、重要な特徴を抽出します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                主成分分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="financial" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>財務分析</CardTitle>
              <CardDescription>キャッシュバーン率、ユニットエコノミクス、成長指標などの財務分析を行います。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                財務分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="market" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>市場・競合分析</CardTitle>
              <CardDescription>市場規模推定、競合マッピング、ポジショニング分析などを行います。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                市場・競合分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="team" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>チーム・組織分析</CardTitle>
              <CardDescription>創業チーム評価、組織成長分析、文化・エンゲージメント強度測定などを行います。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                チーム・組織分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="montecarlo" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>モンテカルロシミュレーション</CardTitle>
              <CardDescription>様々なシナリオに基づいたシミュレーションを実行し、将来予測を行います。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                モンテカルロシミュレーションのコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="sensitivity" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>感度分析</CardTitle>
              <CardDescription>パラメータの変化が出力に与える影響を分析し、トルネードチャートを生成します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                感度分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="portfolio" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>ポートフォリオネットワーク分析</CardTitle>
              <CardDescription>企業間のネットワーク関係を構築・分析し、エコシステム係数を計算します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                ポートフォリオネットワーク分析のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="vc-roi" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>VC向けROI計算</CardTitle>
              <CardDescription>投資収益率の計算と予測を行い、リスク調整済みROIを分析します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                VC向けROI計算のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="health-index" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>健康投資効果指数計算</CardTitle>
              <CardDescription>健康投資効果指数（HIEI）を計算し、エコシステム影響度を分析します。</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] rounded-md bg-background-main p-4 text-center text-text-muted">
                健康投資効果指数計算のコンテンツがここに表示されます
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})

export default { AnalysisContent }

