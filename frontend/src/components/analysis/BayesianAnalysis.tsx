import { useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, HelpCircle } from 'lucide-react';
import { BayesianChart } from '@/components/charts/BayesianChart';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { InsightsContainer, Insight } from '@/components/ui/insights-container';
import { generateBayesianInsights } from '@/lib/ai-insights-generator';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

interface BayesianAnalysisProps {
  companyId: string;
}

// モックのベイズ推論データは削除

type AnalysisType = 'wellnessImpact' | 'retentionRate' | 'revenueGrowth';

export function BayesianAnalysis({ companyId }: BayesianAnalysisProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [analysisType, setAnalysisType] = useState<AnalysisType | null>(null);
  const [data, setData] = useState<any>(null);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [isGeneratingInsights, setIsGeneratingInsights] = useState(false);
  const [activeTab, setActiveTab] = useState<string>('chart');

  // モックデータ：実際の実装では API から取得する
  const mockData = {
    wellnessImpact: {
      title: 'ウェルネス施策の影響分析',
      description: 'ウェルネスプログラムが従業員のエンゲージメントと生産性に与える影響を分析',
      prior: {
        x: Array.from({ length: 50 }, (_, i) => i / 10),
        y: Array.from({ length: 50 }, (_, i) => {
          const x = i / 10;
          return Math.exp(-Math.pow(x - 2.5, 2) / 2) / Math.sqrt(2 * Math.PI);
        }),
        label: '事前分布',
        description: 'プログラム導入前の想定効果：平均5%の生産性向上'
      },
      likelihood: {
        x: Array.from({ length: 50 }, (_, i) => i / 10),
        y: Array.from({ length: 50 }, (_, i) => {
          const x = i / 10;
          return Math.exp(-Math.pow(x - 3.5, 2) / 1.5) / Math.sqrt(1.5 * Math.PI);
        }),
        label: '尤度関数',
        description: '観測データ：平均7%の生産性向上、標準偏差1.2%'
      },
      posterior: {
        x: Array.from({ length: 50 }, (_, i) => i / 10),
        y: Array.from({ length: 50 }, (_, i) => {
          const x = i / 10;
          return Math.exp(-Math.pow(x - 3.0, 2) / 1.2) / Math.sqrt(1.2 * Math.PI);
        }),
        label: '事後分布',
        description: '更新された効果予測：平均7.5%の生産性向上、95%信頼区間は5.1%～9.9%'
      },
      insights: [
        'ウェルネスプログラムの効果は当初予測よりも2.5%高い',
        'エンゲージメント向上への効果が最も顕著で平均9.8%の改善',
        '不確実性（分散）が33%減少し、より確実な予測が可能に',
        'リモートワーカーグループで最も大きな効果（+12.3%）を確認'
      ]
    },
    revenueGrowth: {
      title: '売上成長率のベイジアン予測',
      description: 'ウェルネススコアの向上による売上成長への影響を確率的に分析',
      prior: {
        x: Array.from({ length: 50 }, (_, i) => i / 10),
        y: Array.from({ length: 50 }, (_, i) => {
          const x = i / 10;
          return Math.exp(-Math.pow(x - 2.0, 2) / 2.2) / Math.sqrt(2.2 * Math.PI);
        }),
        label: '事前分布',
        description: '初期予測：ウェルネス向上による売上増加確率62%、平均効果8%'
      },
      likelihood: {
        x: Array.from({ length: 50 }, (_, i) => i / 10),
        y: Array.from({ length: 50 }, (_, i) => {
          const x = i / 10;
          return Math.exp(-Math.pow(x - 3.8, 2) / 1.8) / Math.sqrt(1.8 * Math.PI);
        }),
        label: '尤度関数',
        description: '観測データ：顧客対応部門で18%の売上向上、全体平均12%'
      },
      posterior: {
        x: Array.from({ length: 50 }, (_, i) => i / 10),
        y: Array.from({ length: 50 }, (_, i) => {
          const x = i / 10;
          return Math.exp(-Math.pow(x - 3.2, 2) / 1.4) / Math.sqrt(1.4 * Math.PI);
        }),
        label: '事後分布',
        description: '更新後予測：好影響確率89%に上昇、平均効果12%に上方修正'
      },
      insights: [
        '売上への好影響確率が62%から89%に上昇',
        '特に顧客対応部門での効果が顕著（+18%）',
        'ウェルネススコア1ポイント上昇あたり約2.3%の売上増加',
        '顧客満足度との相関が最も強く、間接的な売上貢献を示唆'
      ]
    },
    programEffectiveness: {
      title: 'ウェルネスプログラム種類別有効性',
      description: '異なるウェルネスプログラム施策の相対的な有効性を確率的に評価',
      prior: {
        x: Array.from({ length: 50 }, (_, i) => i / 10),
        y: Array.from({ length: 50 }, (_, i) => {
          const x = i / 10;
          return Math.exp(-Math.pow(x - 2.5, 2) / 3) / Math.sqrt(3 * Math.PI);
        }),
        label: '事前分布',
        description: '初期予想：すべてのプログラムが同程度の有効性（約65%）と仮定'
      },
      likelihood: {
        x: Array.from({ length: 50 }, (_, i) => i / 10),
        y: Array.from({ length: 50 }, (_, i) => {
          const x = i / 10;
          const y1 = Math.exp(-Math.pow(x - 1.5, 2) / 0.5) / Math.sqrt(0.5 * Math.PI);
          const y2 = Math.exp(-Math.pow(x - 3.5, 2) / 0.8) / Math.sqrt(0.8 * Math.PI);
          const y3 = Math.exp(-Math.pow(x - 4.5, 2) / 0.6) / Math.sqrt(0.6 * Math.PI);
          return (y1 + y2 + y3) / 3;
        }),
        label: '尤度関数',
        description: '実測データ：プログラム間で有効性に大きな差異を確認'
      },
      posterior: {
        x: Array.from({ length: 50 }, (_, i) => i / 10),
        y: Array.from({ length: 50 }, (_, i) => {
          const x = i / 10;
          const y1 = Math.exp(-Math.pow(x - 1.8, 2) / 0.6) / Math.sqrt(0.6 * Math.PI) * 0.12;
          const y2 = Math.exp(-Math.pow(x - 3.2, 2) / 0.7) / Math.sqrt(0.7 * Math.PI) * 0.38;
          const y3 = Math.exp(-Math.pow(x - 4.6, 2) / 0.5) / Math.sqrt(0.5 * Math.PI) * 0.5;
          return y1 + y2 + y3;
        }),
        label: '事後分布',
        description: '更新後評価：柔軟な働き方（92%）、メンタルヘルス（87%）、フィットネス（76%）'
      },
      insights: [
        '柔軟な働き方が最も効果的（有効性92%）',
        'メンタルヘルスサポートが次に効果的（有効性87%）',
        '社内コミュニケーション施策の有効性は予想より低い（54%）',
        '複数のプログラムを組み合わせると総合効果が約15%向上'
      ]
    }
  };

  // 分析実行
  const runAnalysis = useCallback(async () => {
    if (!analysisType) return;

    setIsLoading(true);

    // 実際の実装では API を呼び出す
    await new Promise(resolve => setTimeout(resolve, 1500));

    setData(mockData[analysisType as keyof typeof mockData]);
    setIsLoading(false);
  }, [analysisType]);

  // AIによるインサイト生成
  const generateAiInsights = async () => {
    if (!data || !analysisType) return;

    setIsGeneratingInsights(true);

    try {
      // AI インサイト生成を呼び出す
      const generatedInsights = await generateBayesianInsights(companyId, data);
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
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle>ベイズ分析</CardTitle>
        <CardDescription>
          確率的アプローチで不確実性を定量化し、仮説を更新していきます
        </CardDescription>

        <Accordion type="single" collapsible className="mt-2">
          <AccordionItem value="explanation">
            <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:text-primary transition-colors">
              <span className="flex items-center">
                <HelpCircle className="h-3 w-3 mr-1" />
                ベイズ分析とは？
              </span>
            </AccordionTrigger>
            <AccordionContent className="text-xs text-muted-foreground pb-2">
              <p className="mb-1">
                ベイズ分析は「新しい情報が入るたびに予測を更新する」方法です。単なる予測値だけでなく、その予測が「どのくらい確からしいか」も提供します。事前の知識や仮説（事前分布）に実際のデータ（尤度）を組み合わせて、更新された予測（事後分布）を導き出します。
              </p>
              <p className="mb-1">
                <strong>ビジネス価値：</strong>
                「このウェルネス施策の生産性向上効果は5〜9%（90%信頼区間）」といった具体的な確率を示すことで、より確かな意思決定をサポートします。特にデータが少ない初期段階でも、徐々に精度を高めながら予測できる点が強みです。
              </p>
              <p>
                <strong>分析のポイント：</strong>
                分布の中心（平均）だけでなく、その幅（分散）も重要です。幅が狭いほど確信度が高く、広いほど不確実性が大きいことを示します。
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
                onValueChange={(value) => setAnalysisType(value as AnalysisType)}
              >
                <SelectTrigger id="analysis-type" className="w-full sm:w-[250px]">
                  <SelectValue placeholder="分析タイプを選択" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="wellnessImpact">ウェルネス施策の影響分析</SelectItem>
                  <SelectItem value="revenueGrowth">売上成長率のベイジアン予測</SelectItem>
                  <SelectItem value="programEffectiveness">プログラム種類別有効性</SelectItem>
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
                <TabsTrigger value="chart">チャート</TabsTrigger>
                <TabsTrigger value="insights">インサイト</TabsTrigger>
              </TabsList>

              <TabsContent value="chart" className="space-y-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">{data.title}</CardTitle>
                    <CardDescription>{data.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <BayesianChart
                      priorDistribution={data.prior.samples || []}
                      posteriorDistribution={data.posterior.samples || []}
                      likelihoodPoints={data.likelihood || []}
                    />

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                      <div className="bg-muted/40 rounded-lg p-3">
                        <h4 className="text-sm font-medium text-primary mb-1">{data.prior.label}</h4>
                        <p className="text-xs text-muted-foreground">{data.prior.description}</p>
                      </div>
                      <div className="bg-muted/40 rounded-lg p-3">
                        <h4 className="text-sm font-medium text-primary mb-1">{data.likelihood.label}</h4>
                        <p className="text-xs text-muted-foreground">{data.likelihood.description}</p>
                      </div>
                      <div className="bg-muted/40 rounded-lg p-3">
                        <h4 className="text-sm font-medium text-primary mb-1">{data.posterior.label}</h4>
                        <p className="text-xs text-muted-foreground">{data.posterior.description}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <div className="bg-muted/40 rounded-lg p-4 space-y-2">
                  <h3 className="text-sm font-medium">主要インサイト</h3>
                  <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
                    {data.insights.map((insight: string, index: number) => (
                      <li key={index}>{insight}</li>
                    ))}
                  </ul>
                </div>
              </TabsContent>

              <TabsContent value="insights">
                <InsightsContainer
                  insights={insights}
                  title="ベイジアン分析インサイト"
                  description="確率的アプローチによる分析結果から抽出された重要な知見"
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
  );
}