import { WellnessScoreChart } from '@/components/dashboard/WellnessScoreChart';
import { WellnessMetrics } from '@/components/dashboard/WellnessMetrics';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

// モックデータ
const mockScatterData = [
  { company: 'テックスタート', wellnessScore: 85, growthRate: 45, industry: 'SaaS' },
  { company: 'ヘルスケアイノベーション', wellnessScore: 92, growthRate: 65, industry: 'ヘルスケア' },
  { company: 'グリーンテック', wellnessScore: 78, growthRate: 30, industry: 'クリーンテック' },
  // ... その他のデータポイント
];

export function HomePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[#212121]">ダッシュボード</h1>
        <p className="text-muted-foreground mt-1">
          企業のウェルネスと業績の相関を分析し、投資判断をサポートします
        </p>
      </div>

      <WellnessMetrics
        averageScore={85}
        scoreChange={5}
        topPerformers={8}
        engagementRate={92}
      />

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">概要</TabsTrigger>
          <TabsTrigger value="industry">業界別</TabsTrigger>
          <TabsTrigger value="trends">トレンド</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <WellnessScoreChart data={mockScatterData} />
          
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">主要なインサイト</h3>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-primary">•</span>
                <span>
                  ウェルネススコアが80以上の企業は、平均して45%以上の売上成長率を達成しています
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary">•</span>
                <span>
                  特にSaaS業界では、ウェルネススコアと売上成長率の相関が強く見られます
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary">•</span>
                <span>
                  従業員エンゲージメントが90%を超える企業は、離職率が業界平均の半分以下です
                </span>
              </li>
            </ul>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}