import { useState, useCallback, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, HelpCircle } from 'lucide-react';
import { ScatterPlotChart } from '../charts/ScatterPlotChart';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { mockRegressionData } from '@/mock/regression-data';

interface RegressionAnalysisProps {
  companyId: string;
}

type AnalysisType = 'wellnessVsRevenue' | 'wellnessVsProfit' | 'wellnessVsProductivity';

export function RegressionAnalysis({ companyId }: RegressionAnalysisProps) {
  const [data, setData] = useState<typeof mockRegressionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedAnalysis, setSelectedAnalysis] = useState<AnalysisType>('wellnessVsRevenue');

  // useCallbackでメモ化して不要な再生成を防止
  const runAnalysis = useCallback(async () => {
    setLoading(true);
    try {
      // 実際のAPIが実装されるまではモックデータを使用
      // const response = await fetch(`/api/analysis/regression/${companyId}`);
      // const result = await response.json();

      // モックデータをシミュレーション (遅延を短縮)
      await new Promise(resolve => setTimeout(resolve, 500));
      setData(mockRegressionData);
    } catch (error) {
      console.error('分析の実行中にエラーが発生しました:', error);
    } finally {
      setLoading(false);
    }
  }, [companyId]); // 依存配列に companyId を追加

  // useMemoでデータが変わった時だけ計算を実行
  const insight = useMemo(() => {
    if (!data) return null;

    const currentData = data[selectedAnalysis];
    const r2 = currentData.regressionLine.r2;
    const slope = currentData.regressionLine.slope;

    if (selectedAnalysis === 'wellnessVsRevenue') {
      return {
        heading: 'ウェルネススコアと売上成長率の関係',
        explanation: `ウェルネススコアと売上成長率の間に強い正の相関（R² = ${r2.toFixed(2)}）が見られます。ウェルネススコアが1ポイント上昇すると、売上成長率は平均して約${slope.toFixed(2)}%向上する傾向があります。`
      };
    } else if (selectedAnalysis === 'wellnessVsProfit') {
      return {
        heading: 'ウェルネススコアと利益率の関係',
        explanation: `ウェルネススコアと利益率の間に強い相関（R² = ${r2.toFixed(2)}）があります。ウェルネススコアが1ポイント上がると、利益率は平均して約${slope.toFixed(2)}%増加すると予測されます。`
      };
    } else {
      return {
        heading: 'ウェルネススコアと生産性の関係',
        explanation: `ウェルネススコアと生産性指標の間に非常に強い相関（R² = ${r2.toFixed(2)}）が見られます。ウェルネススコアが1ポイント上昇すると、生産性指標は平均して約${slope.toFixed(1)}ポイント向上します。`
      };
    }
  }, [data, selectedAnalysis]); // 依存配列を適切に設定

  return (
    <Card className="p-4">
      <h3 className="text-lg font-medium mb-2">回帰分析</h3>
      <p className="text-sm text-muted-foreground mb-1">
        ウェルネススコアと経営指標の相関関係を回帰分析で視覚化します
      </p>

      <Accordion type="single" collapsible className="mb-4">
        <AccordionItem value="explanation">
          <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:text-primary transition-colors">
            <span className="flex items-center">
              <HelpCircle className="h-3 w-3 mr-1" />
              回帰分析とは？
            </span>
          </AccordionTrigger>
          <AccordionContent className="text-xs text-muted-foreground pb-2">
            <p className="mb-1">
              回帰分析は2つの要素間の関係性を数値化する手法です。例えば「従業員の健康状態が良くなると売上が増加する」という関係を、
              「ウェルネススコアが1ポイント上がると売上が〇〇%増加する」といった具体的な数字で表します。
            </p>
            <p className="mb-1">
              <strong>ビジネス価値：</strong>
              ウェルネスへの投資が実際の業績にどれだけ影響するかを客観的に示すことで、効果的な経営判断ができます。
            </p>
            <p>
              <strong>分析指標：</strong>
              R²値（0〜1）は相関の強さを示し、1に近いほど強い関係を意味します。傾き（slope）は変化量を表します。
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
          <div className="mb-4">
            <Label htmlFor="analysis-type" className="mb-2 block">分析タイプ</Label>
            <Select
              value={selectedAnalysis}
              onValueChange={(value) => setSelectedAnalysis(value as AnalysisType)}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="分析タイプを選択" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="wellnessVsRevenue">ウェルネス vs 売上成長率</SelectItem>
                <SelectItem value="wellnessVsProfit">ウェルネス vs 利益率</SelectItem>
                <SelectItem value="wellnessVsProductivity">ウェルネス vs 生産性</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <ScatterPlotChart
            data={data[selectedAnalysis].points}
            regressionLine={data[selectedAnalysis].regressionLine}
            xLabel={data[selectedAnalysis].xLabel}
            yLabel={data[selectedAnalysis].yLabel}
          />

          {insight && (
            <div className="mt-4 bg-muted p-3 rounded-md">
              <h4 className="font-medium text-sm">{insight.heading}</h4>
              <p className="text-sm mt-1">{insight.explanation}</p>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}