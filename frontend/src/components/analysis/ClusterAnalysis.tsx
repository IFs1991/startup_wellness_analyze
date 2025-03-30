import { useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, HelpCircle } from 'lucide-react';
import { ClusterChart } from '@/components/charts/ClusterChart';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { InsightsContainer, Insight } from '@/components/ui/insights-container';
import { generateClusterInsights } from '@/lib/ai-insights-generator';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

interface ClusterAnalysisProps {
  companyId: string;
}

interface ClusterData {
  id: string;
  name: string;
  count: number;
  percentage: number;
  color: string;
  features: Record<string, number>;
  insights: string[];
}

interface ClusterPoint {
  x: number;
  y: number;
  clusterId: string;
}

interface AnalysisData {
  title: string;
  description: string;
  clusters: ClusterData[];
  points: ClusterPoint[];
  xAxisLabel: string;
  yAxisLabel: string;
}

// モックのクラスター分析データ
const mockClusterData: Record<string, AnalysisData> = {
  employeeWellness: {
    title: '従業員ウェルネスクラスター',
    description: '従業員のウェルネス特性に基づくセグメンテーション',
    xAxisLabel: 'エンゲージメント',
    yAxisLabel: '満足度',
    clusters: [
      {
        id: 'cluster1',
        name: 'クラスター1: 高エンゲージメント',
        count: 42,
        percentage: 35,
        color: 'rgba(54, 162, 235, 0.7)',
        features: {
          'エンゲージメント': 8.7,
          '満足度': 8.2,
          'ワークライフバランス': 7.5,
          'ストレスレベル': 3.1
        },
        insights: [
          '高いエンゲージメントと満足度を持つグループ',
          'ワークライフバランスの評価も良好',
          '会社の成長に最も貢献している可能性が高いセグメント'
        ]
      },
      {
        id: 'cluster2',
        name: 'クラスター2: バランス志向',
        count: 38,
        percentage: 31,
        color: 'rgba(75, 192, 192, 0.7)',
        features: {
          'エンゲージメント': 6.8,
          '満足度': 7.5,
          'ワークライフバランス': 8.3,
          'ストレスレベル': 3.5
        },
        insights: [
          'ワークライフバランスを特に重視するグループ',
          '満足度は比較的高いが、エンゲージメントは中程度',
          'リモートワークや柔軟な働き方の提供が効果的'
        ]
      },
      {
        id: 'cluster3',
        name: 'クラスター3: ストレス懸念',
        count: 25,
        percentage: 21,
        color: 'rgba(255, 159, 64, 0.7)',
        features: {
          'エンゲージメント': 5.3,
          '満足度': 5.1,
          'ワークライフバランス': 4.8,
          'ストレスレベル': 7.2
        },
        insights: [
          '高いストレスレベルとワークライフバランスの課題を抱えるグループ',
          '満足度とエンゲージメントが低下している',
          'メンタルヘルスサポートと業務負荷の調整が必要'
        ]
      },
      {
        id: 'cluster4',
        name: 'クラスター4: 離職リスク',
        count: 16,
        percentage: 13,
        color: 'rgba(255, 99, 132, 0.7)',
        features: {
          'エンゲージメント': 3.2,
          '満足度': 3.8,
          'ワークライフバランス': 4.1,
          'ストレスレベル': 8.5
        },
        insights: [
          'すべての指標で最も低いスコアを示すグループ',
          '離職リスクが非常に高い',
          '個別面談と具体的な改善アクションの迅速な実施が必要'
        ]
      }
    ],
    points: Array.from({ length: 120 }, () => ({
      x: Math.random() * 10,
      y: Math.random() * 10,
      clusterId: `cluster${Math.floor(Math.random() * 4) + 1}`
    }))
  },
  departmentComparison: {
    title: '部門別ウェルネス比較',
    description: '部門ごとのウェルネス指標とパフォーマンスデータのクラスタリング',
    xAxisLabel: '効率性スコア',
    yAxisLabel: 'ストレスレベル',
    clusters: [
      {
        id: 'dept1',
        name: '低効率・高ストレス部門',
        count: 3,
        percentage: 15,
        color: 'rgba(255, 99, 132, 0.7)',
        features: {
          '平均ウェルネススコア': 4.2,
          '生産性指標': 62,
          '離職率': 28,
          'ストレスレベル': 7.8
        },
        insights: [
          '営業部門と顧客サポート部門がこのクラスターに属している',
          'ストレスレベルが全社平均より35%高い',
          '過剰な目標設定と厳しいKPIが主なストレス要因として報告されている',
          '2四半期連続で目標未達の部門が含まれる'
        ]
      },
      {
        id: 'dept2',
        name: '高効率・低ストレス部門',
        count: 5,
        percentage: 25,
        color: 'rgba(54, 162, 235, 0.7)',
        features: {
          '平均ウェルネススコア': 8.1,
          '生産性指標': 93,
          '離職率': 7,
          'ストレスレベル': 3.2
        },
        insights: [
          'エンジニアリング部門とデザイン部門を含む理想的なクラスター',
          'フレックスタイム制度の活用率が87%と最も高い',
          '週次ウェルネスチェックインの実施率が95%と優れている',
          '過去1年間で生産性が21%向上している'
        ]
      },
      {
        id: 'dept3',
        name: '高効率・高プレッシャー部門',
        count: 4,
        percentage: 20,
        color: 'rgba(255, 206, 86, 0.7)',
        features: {
          '平均ウェルネススコア': 6.5,
          '生産性指標': 88,
          '離職率': 18,
          'ストレスレベル': 6.7
        },
        insights: [
          'プロダクト開発とマーケティング部門がこのクラスターに属している',
          '高い成果を出しているが、締切と競争のプレッシャーも高い',
          '時間外労働が全社平均より32%多い',
          'チームメンバーの42%が「燃え尽き症候群の症状を経験している」と報告'
        ]
      },
      {
        id: 'dept4',
        name: '低効率・組織的課題部門',
        count: 8,
        percentage: 40,
        color: 'rgba(75, 192, 192, 0.7)',
        features: {
          '平均ウェルネススコア': 5.2,
          '生産性指標': 64,
          '離職率': 22,
          'ストレスレベル': 5.7
        },
        insights: [
          '管理部門と財務部門を含むこのクラスターは組織的な課題を抱えている',
          'プロセスとコミュニケーションの問題が主な課題として挙げられている',
          '部門間の連携スコアが最も低い（4.1/10）',
          '直近6ヶ月で生産性が8%低下している'
        ]
      }
    ],
    points: Array.from({ length: 80 }, () => ({
      x: Math.random() * 10,
      y: Math.random() * 10,
      clusterId: `dept${Math.floor(Math.random() * 4) + 1}`
    }))
  }
};

export function ClusterAnalysis({ companyId }: ClusterAnalysisProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [analysisType, setAnalysisType] = useState<string | null>(null);
  const [data, setData] = useState<AnalysisData | null>(null);
  const [selectedCluster, setSelectedCluster] = useState<string | null>(null);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [isGeneratingInsights, setIsGeneratingInsights] = useState(false);
  const [activeTab, setActiveTab] = useState<string>('chart');

  // 分析実行
  const runAnalysis = useCallback(async () => {
    if (!analysisType) return;

    setIsLoading(true);
    setSelectedCluster(null);

    // 実際の実装では API を呼び出す
    await new Promise(resolve => setTimeout(resolve, 1500));

    setData(mockClusterData[analysisType]);
    setIsLoading(false);
  }, [analysisType]);

  // クラスター選択処理
  const handleClusterSelect = (clusterId: string) => {
    setSelectedCluster(clusterId === selectedCluster ? null : clusterId);
  };

  // AIによるインサイト生成
  const generateAiInsights = async () => {
    if (!data || !analysisType) return;

    setIsGeneratingInsights(true);

    try {
      // AI インサイト生成を呼び出す
      const generatedInsights = await generateClusterInsights(companyId, data);
      setInsights(generatedInsights);
    } catch (error) {
      console.error('AIインサイト生成エラー:', error);
    } finally {
      setIsGeneratingInsights(false);
    }
  };

  // インサイトへのフィードバック処理
  const handleInsightFeedback = (insightId: string, isHelpful: boolean) => {
    console.log(`インサイトID: ${insightId}, 役立つ: ${isHelpful}`);
    // 実際の実装ではフィードバックをサーバーに送信
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle>クラスター分析</CardTitle>
          <CardDescription>
            データの類似パターンを発見し、グループ化して表示します
          </CardDescription>

          <Accordion type="single" collapsible className="mt-2">
            <AccordionItem value="explanation">
              <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:text-primary transition-colors">
                <span className="flex items-center">
                  <HelpCircle className="h-3 w-3 mr-1" />
                  クラスター分析とは？
                </span>
              </AccordionTrigger>
              <AccordionContent className="text-xs text-muted-foreground pb-2">
                <p className="mb-1">
                  クラスター分析は「似たもの同士をグループ化する」手法です。例えば、従業員を満足度やエンゲージメントの類似したパターンでグループ分けすることで、特定のニーズや傾向を持つ集団を特定できます。
                </p>
                <p className="mb-1">
                  <strong>ビジネス価値：</strong>
                  例えば「高エンゲージメント・低満足度」という特定の従業員グループが見つかれば、そのグループに特化した改善策を実施できます。また「特定の部門で満足度が極端に低い」など、改善が必要な領域を特定して限られたリソースを効果的に配分できます。
                </p>
                <p>
                  <strong>分析のポイント：</strong>
                  クラスターの大きさ（人数）、特徴（平均値の違い）、そしてクラスター間の距離（類似度）に注目してください。
                </p>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row gap-4 sm:items-end">
              <div className="space-y-1.5">
                <Label htmlFor="analysis-type">分析タイプ</Label>
                <Select
                  value={analysisType || ''}
                  onValueChange={setAnalysisType}
                >
                  <SelectTrigger id="analysis-type" className="w-full sm:w-[250px]">
                    <SelectValue placeholder="分析タイプを選択" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="employeeWellness">従業員ウェルネスクラスター</SelectItem>
                    <SelectItem value="departmentComparison">部門別ウェルネス比較</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button
                onClick={runAnalysis}
                disabled={!analysisType || isLoading}
                className="sm:mb-0.5"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    分析中...
                  </>
                ) : '分析を実行'}
              </Button>
            </div>

            {data && (
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="mb-4">
                  <TabsTrigger value="chart">グラフ</TabsTrigger>
                  <TabsTrigger value="insights">インサイト</TabsTrigger>
                </TabsList>

                <TabsContent value="chart" className="space-y-4">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">{data.title}</CardTitle>
                      <CardDescription>{data.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ClusterChart
                        data={data.points}
                        clusters={data.clusters}
                        selectedCluster={selectedCluster}
                        onClusterSelect={handleClusterSelect}
                        xAxisLabel={data.xAxisLabel}
                        yAxisLabel={data.yAxisLabel}
                        height={400}
                      />
                    </CardContent>
                  </Card>

                  {selectedCluster ? (
                    <div className="space-y-4">
                      {data.clusters
                        .filter(cluster => cluster.id === selectedCluster)
                        .map(cluster => (
                          <Card key={cluster.id} className="overflow-hidden">
                            <div
                              className="h-2"
                              style={{ backgroundColor: cluster.color }}
                            />
                            <CardHeader className="pb-2">
                              <div className="flex justify-between items-start">
                                <CardTitle className="text-base">{cluster.name}</CardTitle>
                                <div className="text-sm text-muted-foreground">
                                  {cluster.count} 名 ({cluster.percentage}%)
                                </div>
                              </div>
                            </CardHeader>
                            <CardContent>
                              <div className="space-y-4">
                                <div>
                                  <h4 className="text-sm font-medium mb-2">主な特徴</h4>
                                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                    {Object.entries(cluster.features).map(([key, value]) => (
                                      <div key={key}>
                                        <div className="flex justify-between text-sm mb-1">
                                          <span>{key}</span>
                                          <span className="font-medium">{value.toFixed(1)}</span>
                                        </div>
                                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                                          <div
                                            className="h-full rounded-full"
                                            style={{
                                              width: `${(value / 10) * 100}%`,
                                              backgroundColor: cluster.color
                                            }}
                                          />
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>

                                <div>
                                  <h4 className="text-sm font-medium mb-2">主要インサイト</h4>
                                  <ul className="space-y-1 text-sm list-disc list-inside">
                                    {cluster.insights.map((insight, index) => (
                                      <li key={index}>{insight}</li>
                                    ))}
                                  </ul>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                    </div>
                  ) : (
                    <div className="bg-muted/40 rounded-lg p-4 text-sm text-center">
                      クラスターをクリックすると詳細情報が表示されます
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="insights">
                  <InsightsContainer
                    insights={insights}
                    title="クラスター分析インサイト"
                    description="クラスター分析から特定されたグループの特徴と洞察"
                    isLoading={isGeneratingInsights}
                    onGenerateAiInsights={generateAiInsights}
                    isGeneratingAiInsights={isGeneratingInsights}
                    onInsightFeedback={handleInsightFeedback}
                  />
                </TabsContent>
              </Tabs>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}