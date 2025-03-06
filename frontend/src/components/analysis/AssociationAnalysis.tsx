import { useState, useCallback, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, HelpCircle } from 'lucide-react';
import { AssociationRulesChart } from '@/components/charts/AssociationRulesChart';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from '@/components/ui/badge';
import { mockAssociationData } from '@/mock/association-data';

interface AssociationAnalysisProps {
  companyId: string;
}

export function AssociationAnalysis({ companyId }: AssociationAnalysisProps) {
  const [data, setData] = useState<typeof mockAssociationData | null>(null);
  const [loading, setLoading] = useState(false);

  // useCallbackでメモ化して不要な再生成を防止
  const runAnalysis = useCallback(async () => {
    setLoading(true);
    try {
      // 実際のAPIが実装されるまではモックデータを使用
      // const response = await fetch(`/api/analysis/association/${companyId}`);
      // const result = await response.json();

      // モックデータをシミュレーション (遅延を短縮)
      await new Promise(resolve => setTimeout(resolve, 500));
      setData(mockAssociationData);
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

  // トップ5のルールのみを表示
  const topRules = useMemo(() => {
    if (!data) return null;
    return [...data.rules]
      .sort((a, b) => b.lift - a.lift)
      .slice(0, 5);
  }, [data]);

  return (
    <Card className="p-4">
      <h3 className="text-lg font-medium mb-2">関連分析</h3>
      <p className="text-sm text-muted-foreground mb-1">
        ウェルネス施策とビジネス成果の間の関連パターンを発見します
      </p>

      <Accordion type="single" collapsible className="mb-4">
        <AccordionItem value="explanation">
          <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:text-primary transition-colors">
            <span className="flex items-center">
              <HelpCircle className="h-3 w-3 mr-1" />
              関連分析とは？
            </span>
          </AccordionTrigger>
          <AccordionContent className="text-xs text-muted-foreground pb-2">
            <p className="mb-1">
              関連分析（アソシエーション分析）は、「AがあるときBも起こる」という関連パターンを発見する手法です。例えば「健康促進プログラムを導入した企業は欠勤率が低下する傾向がある」といったパターンを検出します。
            </p>
            <p className="mb-1">
              <strong>ビジネス価値：</strong>
              どのウェルネス施策が特定のビジネス成果と関連しているかを特定し、効率的な投資判断ができます。複数の施策の組み合わせ効果も発見できます。
            </p>
            <p>
              <strong>主要指標：</strong>
              <span className="block mt-1">
                ・<strong>支持度</strong>：データ全体でそのパターンが発生する割合
                ・<strong>信頼度</strong>：AがあるときにBも発生する条件付き確率
                ・<strong>リフト値</strong>：関連の強さを示す指標（1より大きいほど強い関連）
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
          <div className="flex justify-between mb-4">
            <div className="flex space-x-4">
              <div className="text-sm">
                <div className="text-muted-foreground mb-1">分析されたルール</div>
                <div className="font-semibold text-xl">{data.metrics.totalRules}</div>
              </div>
              <div className="text-sm">
                <div className="text-muted-foreground mb-1">平均信頼度</div>
                <div className="font-semibold text-xl">{(data.metrics.avgConfidence * 100).toFixed(0)}%</div>
              </div>
              <div className="text-sm">
                <div className="text-muted-foreground mb-1">平均リフト値</div>
                <div className="font-semibold text-xl">{data.metrics.avgLift.toFixed(1)}</div>
              </div>
            </div>
            <Badge variant="outline" className="h-fit">信頼度順</Badge>
          </div>

          {topRules && <AssociationRulesChart rules={topRules} height={320} />}

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