import { MetricCard } from './MetricCard';

interface WellnessMetricsProps {
  averageScore: number;
  scoreChange: number;
  topPerformers: number;
  engagementRate: number;
}

export function WellnessMetrics({
  averageScore,
  scoreChange,
  topPerformers,
  engagementRate
}: WellnessMetricsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <MetricCard
        title="平均ウェルネススコア"
        value={averageScore}
        change={scoreChange}
        trend={scoreChange >= 0 ? 'up' : 'down'}
        description="全企業の平均スコア"
      />
      <MetricCard
        title="スコア上位企業"
        value={topPerformers}
        description="ウェルネススコア80以上の企業数"
      />
      <MetricCard
        title="従業員エンゲージメント"
        value={`${engagementRate}%`}
        description="アンケート回答率の平均"
      />
      <MetricCard
        title="分析対象企業"
        value="23"
        change={2}
        trend="up"
        description="先月比"
      />
    </div>
  );
}