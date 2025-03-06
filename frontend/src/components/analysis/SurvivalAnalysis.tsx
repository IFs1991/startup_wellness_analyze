import { useState, useCallback, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, HelpCircle } from 'lucide-react';
import { SurvivalCurveChart } from '@/components/charts/SurvivalCurveChart';
import { HazardRateChart } from '@/components/charts/HazardRateChart';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { mockSurvivalData } from '@/mock/survival-data';

interface SurvivalAnalysisProps {
  companyId: string;
}

export function SurvivalAnalysis({ companyId }: SurvivalAnalysisProps) {
  const [data, setData] = useState<typeof mockSurvivalData | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('curves');

  // useCallbackでメモ化して不要な再生成を防止
  const runAnalysis = useCallback(async () => {
    setLoading(true);
    try {
      // 実際のAPIが実装されるまではモックデータを使用
      // const response = await fetch(`/api/analysis/survival/${companyId}`);
      // const result = await response.json();

      // モックデータをシミュレーション (遅延を短縮)
      await new Promise(resolve => setTimeout(resolve, 500));
      setData(mockSurvivalData);
    } catch (error) {
      console.error('分析の実行中にエラーが発生しました:', error);
    } finally {
      setLoading(false);
    }
  }, [companyId]);

  // useMemoでデータが変わった時だけ計算を実行
  const insights = useMemo(() => {
    if (!data) return null;
    return data.insights;
  }, [data]);

  return (
    <Card className="p-4">
      <h3 className="text-lg font-medium mb-2">生存分析</h3>
      <p className="text-sm text-muted-foreground mb-1">
        従業員の定着率と離職リスク要因を時間経過とともに分析します
      </p>

      <Accordion type="single" collapsible className="mb-4">
        <AccordionItem value="explanation">
          <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:text-primary transition-colors">
            <span className="flex items-center">
              <HelpCircle className="h-3 w-3 mr-1" />
              生存分析とは？
            </span>
          </AccordionTrigger>
          <AccordionContent className="text-xs text-muted-foreground pb-2">
            <p className="mb-1">
              生存分析（サバイバル分析）は、特定のイベント（例：従業員の離職）が発生するまでの時間を分析する統計手法です。
              「どのような要因が離職リスクを高めるか」「どの時点で離職リスクが高まるか」などを定量的に評価できます。
            </p>
            <p className="mb-1">
              <strong>ビジネス価値：</strong>
              従業員の定着率向上のための効果的な介入タイミングや対象グループを特定できます。
              また、ウェルネス施策が従業員の定着にどれだけ貢献しているかを客観的に評価できます。
            </p>
            <p>
              <strong>主要指標：</strong>
              <span className="block mt-1">
                ・<strong>生存曲線</strong>：時間経過に伴う定着率の変化を示すグラフ
                ・<strong>ハザード率</strong>：特定の時点での離職リスク
                ・<strong>ハザード比</strong>：各要因が離職リスクに与える影響の大きさ
              </span>
            </p>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      {!data && (
        <Button onClick={runAnalysis} disabled={loading}>
          {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          分析を実行
        </Button>
      )}

      {data && (
        <div className="mt-4">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="curves">生存曲線</TabsTrigger>
              <TabsTrigger value="hazard">ハザード率</TabsTrigger>
              <TabsTrigger value="factors">リスク要因</TabsTrigger>
            </TabsList>

            <TabsContent value="curves" className="pt-4">
              <SurvivalCurveChart curves={data.survivalCurves} height={350} />
              <div className="mt-4 grid grid-cols-3 gap-3">
                <div className="bg-muted p-3 rounded-md">
                  <div className="text-xs text-muted-foreground">全体の中央生存時間</div>
                  <div className="font-semibold text-xl">{data.medianSurvivalTime.overall}ヶ月</div>
                </div>
                <div className="bg-muted p-3 rounded-md">
                  <div className="text-xs text-muted-foreground">ウェルネス高グループ</div>
                  <div className="font-semibold text-xl text-green-600">{data.medianSurvivalTime.highWellness}ヶ月</div>
                </div>
                <div className="bg-muted p-3 rounded-md">
                  <div className="text-xs text-muted-foreground">ウェルネス低グループ</div>
                  <div className="font-semibold text-xl text-red-600">{data.medianSurvivalTime.lowWellness}ヶ月</div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="hazard" className="pt-4">
              <HazardRateChart data={data.hazardRates} height={300} />
              <div className="mt-4 text-sm">
                <p className="text-muted-foreground">
                  ハザード率は各時点での離職リスクを表します。値が高いほど、その期間中に離職する確率が高くなります。
                  信頼度の低い期間（点線）は予測値であり、実際のデータが少ないため不確実性が高くなります。
                </p>
              </div>
            </TabsContent>

            <TabsContent value="factors" className="pt-4">
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-muted">
                      <th className="text-left p-2 border">リスク要因</th>
                      <th className="text-center p-2 border">ハザード比</th>
                      <th className="text-center p-2 border">p値</th>
                      <th className="text-center p-2 border">影響度</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.riskFactors.map((factor, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-muted/30'}>
                        <td className="p-2 border">{factor.factor}</td>
                        <td className="text-center p-2 border font-semibold">{factor.hazardRatio.toFixed(2)}</td>
                        <td className="text-center p-2 border">{factor.pValue.toFixed(3)}</td>
                        <td className="p-2 border">
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div
                              className="h-2.5 rounded-full bg-red-600"
                              style={{ width: `${Math.min(100, factor.hazardRatio / 3 * 100)}%` }}
                            ></div>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-3 text-xs text-muted-foreground">
                <p>
                  <strong>ハザード比</strong>：値が大きいほど離職リスクへの影響が大きい（1.0は影響なし）
                </p>
                <p>
                  <strong>p値</strong>：0.05未満で統計的に有意な関連性あり
                </p>
              </div>
            </TabsContent>
          </Tabs>

          <div className="mt-6 text-sm">
            <h4 className="font-medium">主要な発見：</h4>
            <ul className="list-disc pl-5 mt-2">
              {insights?.map((insight, i) => (
                <li key={i} className="mb-1">{insight}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </Card>
  );
}