import { useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, HelpCircle } from 'lucide-react';
import { TimeSeriesChart, TimeSeriesData } from '@/components/charts/TimeSeriesChart';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger
} from '@/components/ui/tabs';
import { InsightsContainer, Insight } from '@/components/ui/insights-container';
import { generateTimeSeriesInsights } from '@/lib/ai-insights-generator';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

interface TimeSeriesAnalysisProps {
  companyId: string;
}

// モックの時系列分析データ
const mockTimeSeriesData = {
  wellnessScoreTrend: {
    title: 'ウェルネススコアの時系列推移',
    description: '過去12ヶ月間のウェルネススコアとその構成要素の推移',
    data: {
      dates: [
        '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
        '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12'
      ],
      series: [
        {
          name: '総合スコア',
          data: [72, 73, 75, 74, 76, 78, 79, 81, 82, 83, 82, 84],
          color: 'rgba(75, 192, 192, 1)',
          isMainMetric: true
        },
        {
          name: 'エンゲージメント',
          data: [70, 72, 73, 72, 75, 77, 80, 82, 83, 84, 82, 85],
          color: 'rgba(54, 162, 235, 0.6)',
          isMainMetric: false
        },
        {
          name: '満足度',
          data: [75, 74, 76, 75, 77, 79, 80, 81, 82, 83, 81, 84],
          color: 'rgba(153, 102, 255, 0.6)',
          isMainMetric: false
        },
        {
          name: 'ワークライフバランス',
          data: [68, 70, 72, 73, 75, 76, 77, 79, 80, 81, 80, 82],
          color: 'rgba(255, 159, 64, 0.6)',
          isMainMetric: false
        }
      ],
      annotations: [
        {
          date: '2023-06',
          text: 'リモートワークポリシー導入',
          impact: 'positive' as const
        },
        {
          date: '2023-09',
          text: 'ウェルネス研修実施',
          impact: 'positive' as const
        },
        {
          date: '2023-11',
          text: '組織再編',
          impact: 'negative' as const
        }
      ],
      insights: [
        'ウェルネススコアは過去12ヶ月で12ポイント（16.7%）上昇',
        'リモートワークポリシー導入後、特にワークライフバランスの改善が顕著',
        '組織再編後に一時的な低下が見られたが、その後回復',
        'エンゲージメントが総合スコアの中で最も高い成長を示している'
      ]
    }
  },
  wellnessFinancialCorrelation: {
    title: 'ウェルネスと財務指標の相関推移',
    description: 'ウェルネススコアと主要財務指標の時系列相関',
    data: {
      dates: [
        '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
        '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12'
      ],
      series: [
        {
          name: 'ウェルネススコア',
          data: [72, 73, 75, 74, 76, 78, 79, 81, 82, 83, 82, 84],
          color: 'rgba(75, 192, 192, 1)',
          isMainMetric: true
        },
        {
          name: '生産性指標',
          data: [68, 70, 73, 74, 77, 80, 82, 85, 87, 88, 86, 90],
          color: 'rgba(54, 162, 235, 0.6)',
          isMainMetric: false
        },
        {
          name: '売上成長率',
          data: [3.2, 3.3, 3.5, 3.7, 4.0, 4.2, 4.5, 4.8, 5.1, 5.4, 5.2, 5.6],
          scale: 15, // 表示スケール調整
          color: 'rgba(255, 99, 132, 0.6)',
          isMainMetric: false
        },
        {
          name: '利益率',
          data: [12.5, 12.8, 13.2, 13.5, 14.1, 14.8, 15.4, 16.0, 16.8, 17.5, 17.2, 18.1],
          scale: 5, // 表示スケール調整
          color: 'rgba(255, 159, 64, 0.6)',
          isMainMetric: false
        }
      ],
      annotations: [
        {
          date: '2023-06',
          text: 'ウェルネスプログラム拡充',
          impact: 'positive' as const
        },
        {
          date: '2023-09',
          text: '四半期業績発表',
          impact: 'positive' as const
        }
      ],
      insights: [
        'ウェルネススコアと生産性指標の間に強い相関（相関係数: 0.91）',
        'ウェルネススコアが1ポイント上昇するごとに、売上成長率は約0.2%向上',
        'ウェルネススコアの改善が利益率の向上に1-2ヶ月遅れて反映される傾向',
        'ウェルネスプログラム拡充後、全指標の上昇率が加速'
      ]
    }
  },
  surveyResponseTrend: {
    title: '従業員調査回答の時系列推移',
    description: '四半期ごとの従業員調査回答率と平均スコアの推移',
    data: {
      dates: ['2022-Q1', '2022-Q2', '2022-Q3', '2022-Q4', '2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4'],
      series: [
        {
          name: '回答率',
          data: [65, 68, 72, 75, 78, 83, 87, 92],
          color: 'rgba(75, 192, 192, 1)',
          isMainMetric: true
        },
        {
          name: '満足度スコア',
          data: [68, 70, 72, 74, 76, 79, 81, 84],
          color: 'rgba(54, 162, 235, 0.6)',
          isMainMetric: false
        },
        {
          name: '信頼度指標',
          data: [65, 67, 70, 73, 76, 80, 83, 85],
          color: 'rgba(153, 102, 255, 0.6)',
          isMainMetric: false
        },
        {
          name: '推奨度スコア',
          data: [62, 65, 69, 72, 75, 79, 82, 86],
          color: 'rgba(255, 159, 64, 0.6)',
          isMainMetric: false
        }
      ],
      annotations: [
        {
          date: '2022-Q3',
          text: '匿名フィードバック導入',
          impact: 'positive' as const
        },
        {
          date: '2023-Q1',
          text: 'アクションプラン実施',
          impact: 'positive' as const
        },
        {
          date: '2023-Q3',
          text: '調査プロセス改善',
          impact: 'positive' as const
        }
      ],
      insights: [
        '回答率は2年間で27ポイント（41.5%）上昇し、より正確なデータ収集が可能に',
        '匿名フィードバック導入後、全スコアの上昇率が加速',
        '調査に基づくアクションプランの実施が信頼度指標向上に貢献',
        '回答率の向上と満足度スコアの間に正の相関が見られる'
      ]
    }
  }
};

type AnalysisType = 'wellnessScoreTrend' | 'wellnessFinancialCorrelation' | 'surveyResponseTrend';

export function TimeSeriesAnalysis({ companyId }: TimeSeriesAnalysisProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [analysisType, setAnalysisType] = useState<AnalysisType | null>(null);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [isGeneratingInsights, setIsGeneratingInsights] = useState(false);
  const [activeTab, setActiveTab] = useState<string>('chart');

  const [data, setData] = useState<TimeSeriesData | null>(null);

  const runAnalysis = useCallback(async () => {
    setIsLoading(true);

    // 実際の実装では API を呼び出す
    await new Promise(resolve => setTimeout(resolve, 1500));

    if (analysisType === 'wellnessScoreTrend') {
      setData(mockTimeSeriesData.wellnessScoreTrend.data);
    } else if (analysisType === 'wellnessFinancialCorrelation') {
      setData(mockTimeSeriesData.wellnessFinancialCorrelation.data);
    } else if (analysisType === 'surveyResponseTrend') {
      setData(mockTimeSeriesData.surveyResponseTrend.data);
    }

    setIsLoading(false);
  }, [analysisType]);

  // AIによるインサイト生成
  const generateAiInsights = async () => {
    if (!data || !analysisType) return;

    setIsGeneratingInsights(true);

    try {
      // AI インサイト生成を呼び出す
      const generatedInsights = await generateTimeSeriesInsights(companyId, data);
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
        <CardTitle>時系列分析</CardTitle>
        <CardDescription>
          時間経過に伴う変化パターンとトレンドを可視化します
        </CardDescription>

        <Accordion type="single" collapsible className="mt-2">
          <AccordionItem value="explanation">
            <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:text-primary transition-colors">
              <span className="flex items-center">
                <HelpCircle className="h-3 w-3 mr-1" />
                時系列分析とは？
              </span>
            </AccordionTrigger>
            <AccordionContent className="text-xs text-muted-foreground pb-2">
              <p className="mb-1">
                時系列分析は「時間の経過に伴う変化パターン」を見つける手法です。長期的な成長トレンド、季節変動、特定のイベントによる影響などを識別します。
              </p>
              <p className="mb-1">
                <strong>ビジネス価値：</strong>
                「過去6ヶ月間の従業員満足度が徐々に低下している」といったトレンドを早期に発見することで、問題が深刻化する前に対策を打てます。また「ウェルネスプログラム導入後に生産性が向上した」といった施策効果も定量的に確認できます。
              </p>
              <p>
                <strong>分析のポイント：</strong>
                全体的なトレンド（上昇/下降）、変化のタイミング（急激な変化点）、定期的なパターン（季節性）、そして注釈されたイベントとの関連性に注目してください。
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
                  <SelectItem value="wellnessScoreTrend">ウェルネススコア推移</SelectItem>
                  <SelectItem value="wellnessFinancialCorrelation">ウェルネスと財務指標の相関</SelectItem>
                  <SelectItem value="surveyResponseTrend">調査回答率と結果の推移</SelectItem>
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
                  <CardContent className="pt-6">
                    <TimeSeriesChart data={data} />
                  </CardContent>
                </Card>

                {data && (
                  <>
                    <div className="bg-muted/40 rounded-lg p-4 space-y-2">
                      <h3 className="text-sm font-medium">主要インサイト</h3>
                      <ul className="space-y-1 text-sm text-muted-foreground list-disc list-inside">
                        {data.insights.map((insight: string, index: number) => (
                          <li key={index}>{insight}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="bg-muted/40 rounded-lg p-4 space-y-2">
                      <h3 className="text-sm font-medium">注釈</h3>
                      <div className="space-y-3">
                        {data.annotations.map((annotation, index: number) => (
                          <div key={index} className="flex space-x-3 text-sm">
                            <div className="font-medium min-w-[80px]">{annotation.date}</div>
                            <div className={`flex-1 ${
                              annotation.impact === 'positive' ? 'text-green-600' :
                              annotation.impact === 'negative' ? 'text-red-600' : 'text-gray-600'
                            }`}>
                              {annotation.text}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </TabsContent>

              <TabsContent value="insights">
                <InsightsContainer
                  insights={insights}
                  title="時系列分析インサイト"
                  description="時系列データから抽出された重要なパターン、傾向、相関関係"
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