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
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

interface CorrelationAnalysisProps {
  companyId: string;
}

// モック相関データ
const correlationData = [
  { item: 'ウェルネススコア', salesGrowth: 0.85, profitMargin: 0.72, employees: 0.45 },
  { item: 'エンゲージメント', salesGrowth: 0.78, profitMargin: 0.65, employees: 0.32 },
  { item: '満足度', salesGrowth: 0.82, profitMargin: 0.69, employees: 0.40 },
  { item: 'ワークライフバランス', salesGrowth: 0.71, profitMargin: 0.68, employees: 0.30 },
];

export function CorrelationAnalysis({ companyId }: CorrelationAnalysisProps) {
  const [loading, setLoading] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);

  // 相関分析の実行（実際はAPIから取得）
  const runAnalysis = useCallback(async () => {
    setLoading(true);
    try {
      // 実際のAPIが実装されるまではモックデータを使用
      await new Promise(resolve => setTimeout(resolve, 500));
    } catch (error) {
      console.error('分析の実行中にエラーが発生しました:', error);
    } finally {
      setLoading(false);
    }
  }, [companyId]);

  return (
    <div>
      <div className="mb-4">
        <h2 className="text-xl font-bold mb-2">相関分析</h2>
        <p className="text-sm text-muted-foreground">ウェルネススコアと財務指標の相関関係を分析します</p>
      </div>

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

      <div className="mt-6">
        <h3 className="font-semibold mb-4">相関係数行列</h3>
        <div className="overflow-x-auto bg-white rounded-lg p-4 shadow-sm">
          <table className="min-w-full border-collapse">
            <thead>
              <tr>
                <th className="px-4 py-2 border bg-muted/50 text-left"></th>
                <th className="px-4 py-2 border bg-muted/50 text-center">売上成長率</th>
                <th className="px-4 py-2 border bg-muted/50 text-center">利益率</th>
                <th className="px-4 py-2 border bg-muted/50 text-center">従業員数</th>
              </tr>
            </thead>
            <tbody>
              {correlationData.map((row, index) => (
                <tr key={index}>
                  <td className="px-4 py-2 border font-medium">{row.item}</td>
                  <td className="px-4 py-2 border text-center" style={{
                    backgroundColor: `rgba(0, 0, 128, ${row.salesGrowth})`,
                    color: row.salesGrowth > 0.5 ? 'white' : 'black'
                  }}>
                    {row.salesGrowth.toFixed(2)}
                  </td>
                  <td className="px-4 py-2 border text-center" style={{
                    backgroundColor: `rgba(0, 0, 128, ${row.profitMargin})`,
                    color: row.profitMargin > 0.5 ? 'white' : 'black'
                  }}>
                    {row.profitMargin.toFixed(2)}
                  </td>
                  <td className="px-4 py-2 border text-center" style={{
                    backgroundColor: `rgba(0, 0, 128, ${row.employees})`,
                    color: row.employees > 0.5 ? 'white' : 'black'
                  }}>
                    {row.employees.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-6">
        <h3 className="font-semibold mb-4">グラフ表示</h3>
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="w-full aspect-[4/3]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={correlationData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="item" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="salesGrowth" stroke="#8884d8" name="売上成長率" />
                <Line type="monotone" dataKey="profitMargin" stroke="#82ca9d" name="利益率" />
                <Line type="monotone" dataKey="employees" stroke="#ffc658" name="従業員" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="mt-6 p-4 bg-blue-50 border border-blue-100 rounded-lg">
        <h4 className="font-semibold mb-2">分析結果：</h4>
        <p className="text-sm">
          従業員のウェルネススコアと売上成長率の間に強い相関（0.85）が見られます。また、エンゲージメントと売上成長率の間にも強い相関（0.78）があります。これらの結果から、従業員のウェルビーイングへの投資が事業成果の向上につながる可能性が示唆されます。
        </p>
      </div>
    </div>
  );
}