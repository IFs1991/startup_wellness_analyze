import { useState, useCallback, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, HelpCircle } from 'lucide-react';
import { CorrelationMatrix } from '@/components/charts/CorrelationMatrix';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { mockCorrelationData } from '@/mock/correlation-data';

interface CorrelationAnalysisProps {
  companyId: string;
}

export function CorrelationAnalysis({ companyId }: CorrelationAnalysisProps) {
  const [data, setData] = useState<typeof mockCorrelationData | null>(null);
  const [loading, setLoading] = useState(false);

  // useCallbackでメモ化して不要な再生成を防止
  const runAnalysis = useCallback(async () => {
    setLoading(true);
    try {
      // 実際のAPIが実装されるまではモックデータを使用
      // const response = await fetch(`/api/analysis/correlation/${companyId}`);
      // const result = await response.json();

      // モックデータをシミュレーション (遅延を短縮)
      await new Promise(resolve => setTimeout(resolve, 500));
      setData(mockCorrelationData);
    } catch (error) {
      console.error('分析の実行中にエラーが発生しました:', error);
    } finally {
      setLoading(false);
    }
  }, [companyId]); // 依存配列に companyId を追加

  // useMemoでデータが変わった時だけ計算を実行
  const insights = useMemo(() => {
    if (!data) return null;
    return data.insights;
  }, [data]);

  return (
    <Card className="p-4">
      <h3 className="text-lg font-medium mb-2">相関分析</h3>
      <p className="text-sm text-muted-foreground mb-1">
        ウェルネススコアと財務指標の相関関係を分析します
      </p>

      <Accordion type="single" collapsible className="mb-4">
        <AccordionItem value="explanation">
          <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:text-primary transition-colors">
            <span className="flex items-center">
              <HelpCircle className="h-3 w-3 mr-1" />
              相関分析とは？
            </span>
          </AccordionTrigger>
          <AccordionContent className="text-xs text-muted-foreground pb-2">
            <p className="mb-1">
              相関分析は「2つの指標の間に関連性があるか」を測定する手法です。例えば、従業員の満足度と会社の売上の間に関連性があるかを-1から1の数値で示します。1に近いほど強い正の相関（一方が上がると他方も上がる）、-1に近いほど強い負の相関（一方が上がると他方は下がる）を表します。
            </p>
            <p className="mb-1">
              <strong>ビジネス価値：</strong>
              「従業員のエンゲージメントと顧客満足度の間に強い相関（0.78）がある」といった発見は、従業員満足度向上への投資が顧客体験の向上にもつながる可能性を示します。これにより、限られたリソースを最も効果的な領域に集中できます。
            </p>
            <p>
              <strong>注意点：</strong>
              相関は因果関係を必ずしも意味しません。相関が見つかったら、その関係をより深く調査することが重要です。
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
          <CorrelationMatrix data={data.correlations} />
          <div className="mt-4 text-sm">
            <h4 className="font-medium">主要な発見：</h4>
            <ul className="list-disc pl-5 mt-2">
              {insights?.map((insight, i) => (
                <li key={i}>{insight}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </Card>
  );
}