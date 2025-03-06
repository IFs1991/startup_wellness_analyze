import { useState, useCallback, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, HelpCircle } from 'lucide-react';
import { WordCloudChart } from '@/components/charts/WordCloudChart';
import { SentimentGauge } from '@/components/charts/SentimentGauge';
import { TopicBarChart } from '@/components/charts/TopicBarChart';
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
import { mockTextMiningData } from '@/mock/text-mining-data';

interface TextMiningAnalysisProps {
  companyId: string;
}

export function TextMiningAnalysis({ companyId }: TextMiningAnalysisProps) {
  const [data, setData] = useState<typeof mockTextMiningData | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('wordcloud');

  // useCallbackでメモ化して不要な再生成を防止
  const runAnalysis = useCallback(async () => {
    setLoading(true);
    try {
      // 実際のAPIが実装されるまではモックデータを使用
      // const response = await fetch(`/api/analysis/text-mining/${companyId}`);
      // const result = await response.json();

      // モックデータをシミュレーション (遅延を短縮)
      await new Promise(resolve => setTimeout(resolve, 500));
      setData(mockTextMiningData);
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
      <h3 className="text-lg font-medium mb-2">テキストマイニング分析</h3>
      <p className="text-sm text-muted-foreground mb-1">
        フィードバックや社内コミュニケーションからパターンと感情を抽出します
      </p>

      <Accordion type="single" collapsible className="mb-4">
        <AccordionItem value="explanation">
          <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:text-primary transition-colors">
            <span className="flex items-center">
              <HelpCircle className="h-3 w-3 mr-1" />
              テキストマイニングとは？
            </span>
          </AccordionTrigger>
          <AccordionContent className="text-xs text-muted-foreground pb-2">
            <p className="mb-1">
              テキストマイニングは、大量のテキストデータから有用なパターンや知識を抽出する技術です。従業員のフィードバック、
              社内コミュニケーション、アンケート回答などのテキストから情報を発見します。
            </p>
            <p className="mb-1">
              <strong>ビジネス価値：</strong>
              従業員の意見や感情を可視化し、組織内の主要な関心事やトレンドを特定できます。
              また、ウェルネス施策の効果や従業員のニーズを把握することで、より効果的なプログラムの設計が可能になります。
            </p>
            <p>
              <strong>主要手法：</strong>
              <span className="block mt-1">
                ・<strong>ワードクラウド</strong>：頻出キーワードの視覚化
                ・<strong>感情分析</strong>：テキストから感情（ポジティブ/ネガティブ）を抽出
                ・<strong>トピックモデリング</strong>：テキスト内の主要トピックを自動検出
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
              <TabsTrigger value="wordcloud">ワードクラウド</TabsTrigger>
              <TabsTrigger value="sentiment">感情分析</TabsTrigger>
              <TabsTrigger value="topics">トピック分析</TabsTrigger>
            </TabsList>

            <TabsContent value="wordcloud" className="pt-4">
              <WordCloudChart data={data.wordCloud} height={350} />
              <div className="mt-4 text-sm">
                <h4 className="font-medium mb-2">最も頻出するキーワード:</h4>
                <div className="grid grid-cols-5 gap-2">
                  {data.wordCloud.slice(0, 5).map((word, i) => (
                    <div key={i} className="bg-muted p-2 rounded-md">
                      <div className="font-semibold">{word.text}</div>
                      <div className="text-xs text-muted-foreground">スコア: {word.value}</div>
                    </div>
                  ))}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="sentiment" className="pt-4">
              <div className="flex justify-center mb-6">
                <SentimentGauge
                  positive={data.sentimentAnalysis.positive}
                  negative={data.sentimentAnalysis.negative}
                  neutral={data.sentimentAnalysis.neutral}
                  size={250}
                />
              </div>
              <div className="grid grid-cols-3 gap-3">
                {data.keyTerms.map((term, i) => (
                  <div key={i} className="bg-muted p-3 rounded-md">
                    <div className="font-semibold">{term.term}</div>
                    <div className="text-xs text-muted-foreground mt-1">出現回数: {term.occurrence}</div>
                    <div className="mt-1 h-2 w-full bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${term.sentiment * 100}%`,
                          backgroundColor: `hsl(${term.sentiment * 120}, 70%, 50%)`
                        }}
                      />
                    </div>
                    <div className="text-xs text-right mt-1">感情スコア: {term.sentiment.toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="topics" className="pt-4">
              <TopicBarChart topics={data.topicModeling} height={300} />
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