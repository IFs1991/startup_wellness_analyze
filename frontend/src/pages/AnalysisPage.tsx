import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MonthlyTrendChart } from '@/components/charts/MonthlyTrendChart';
import { TimeSeriesChart } from '@/components/charts/TimeSeriesChart';
import { SentimentGauge } from '@/components/charts/SentimentGauge';
import { SurvivalCurveChart } from '@/components/charts/SurvivalCurveChart';
import { WordCloudChart } from '@/components/charts/WordCloudChart';
import { TopicBarChart } from '@/components/charts/TopicBarChart';
import { ClusterChart } from '@/components/charts/ClusterChart';
import { CorrelationMatrix } from '@/components/charts/CorrelationMatrix';
import { BayesianAnalysis } from '@/components/analysis/BayesianAnalysis';
import { fadeInAnimation, slideInAnimation, zoomInAnimation } from '@/lib/utils';

const monthlyData = [
  { name: '1月', value: 400 },
  { name: '2月', value: 300 },
  { name: '3月', value: 600 },
  { name: '4月', value: 800 },
  { name: '5月', value: 500 },
];

// 生存曲線のデモデータ
const survivalCurves = [
  {
    name: 'グループA',
    color: '#4f46e5',
    data: [
      { time: 0, probability: 1.0 },
      { time: 6, probability: 0.95 },
      { time: 12, probability: 0.85 },
      { time: 18, probability: 0.78 },
      { time: 24, probability: 0.72 }
    ]
  },
  {
    name: 'グループB',
    color: '#ef4444',
    data: [
      { time: 0, probability: 1.0 },
      { time: 6, probability: 0.88 },
      { time: 12, probability: 0.76 },
      { time: 18, probability: 0.62 },
      { time: 24, probability: 0.55 }
    ]
  }
];

// 時系列データのデモデータ
const timeSeriesData = {
  dates: Array.from({ length: 50 }, (_, i) => new Date(2023, 0, i + 1).toISOString().split('T')[0]),
  series: [
    {
      name: 'ウェルネススコア',
      data: Array.from({ length: 50 }, (_, i) => Math.sin(i * 0.1) * 10 + 20 + Math.random() * 5),
      color: '#4f46e5',
      isMainMetric: true
    }
  ],
  annotations: [],
  insights: ['ウェルネススコアは緩やかな上昇傾向にあります。', '定期的な波形パターンが見られます。']
};

// ワードクラウドのデモデータ
const wordCloudData = [
  { text: 'ワークライフバランス', value: 30 },
  { text: 'リモートワーク', value: 25 },
  { text: 'ストレス', value: 20 },
  { text: '残業', value: 18 },
  { text: '休暇', value: 15 },
  { text: '給与', value: 12 },
  { text: '満足度', value: 10 },
  { text: '評価', value: 8 }
];

// トピック分析のデモデータ
const topicData = [
  { topic: 'ワークスタイル', keywords: ['リモート', '柔軟', '時間'], documentCount: 32, sentiment: 0.7 },
  { topic: '評価制度', keywords: ['公平', '透明', '実績'], documentCount: 28, sentiment: 0.2 },
  { topic: '福利厚生', keywords: ['健康', '保険', '休暇'], documentCount: 24, sentiment: 0.8 },
  { topic: 'オフィス環境', keywords: ['快適', '設備', '空間'], documentCount: 18, sentiment: 0.5 },
  { topic: 'コミュニケーション', keywords: ['会議', '情報', '共有'], documentCount: 14, sentiment: 0.1 }
];

// 相関データ
const correlationData = [
  { x: '満足度', y: '満足度', value: 1.0 },
  { x: '満足度', y: '生産性', value: 0.7 },
  { x: '満足度', y: 'ストレス', value: -0.5 },
  { x: '満足度', y: '離職率', value: -0.6 },
  { x: '生産性', y: '満足度', value: 0.7 },
  { x: '生産性', y: '生産性', value: 1.0 },
  { x: '生産性', y: 'ストレス', value: -0.3 },
  { x: '生産性', y: '離職率', value: -0.4 },
  { x: 'ストレス', y: '満足度', value: -0.5 },
  { x: 'ストレス', y: '生産性', value: -0.3 },
  { x: 'ストレス', y: 'ストレス', value: 1.0 },
  { x: 'ストレス', y: '離職率', value: 0.5 },
  { x: '離職率', y: '満足度', value: -0.6 },
  { x: '離職率', y: '生産性', value: -0.4 },
  { x: '離職率', y: 'ストレス', value: 0.5 },
  { x: '離職率', y: '離職率', value: 1.0 }
];

const AnalysisPage: React.FC = () => {
  const [selectedCluster, setSelectedCluster] = useState<string | null>(null);

  const clusterData = {
    points: [
      { x: 10, y: 20, clusterId: '1', label: '企業1' },
      { x: 15, y: 25, clusterId: '1', label: '企業2' },
      { x: 12, y: 18, clusterId: '1', label: '企業3' },
      { x: 30, y: 40, clusterId: '2', label: '企業4' },
      { x: 32, y: 45, clusterId: '2', label: '企業5' },
      { x: 28, y: 38, clusterId: '2', label: '企業6' },
      { x: 50, y: 30, clusterId: '3', label: '企業7' },
      { x: 55, y: 32, clusterId: '3', label: '企業8' },
      { x: 48, y: 28, clusterId: '3', label: '企業9' }
    ],
    clusters: [
      {
        id: '1',
        name: 'グループA',
        color: '#4f46e5',
        count: 3,
        percentage: 33.3,
        features: { 満足度: 85, 生産性: 78 },
        insights: ['高い満足度と安定した生産性']
      },
      {
        id: '2',
        name: 'グループB',
        color: '#7c3aed',
        count: 3,
        percentage: 33.3,
        features: { 満足度: 65, 生産性: 88 },
        insights: ['高い生産性だが満足度は中程度']
      },
      {
        id: '3',
        name: 'グループC',
        color: '#ef4444',
        count: 3,
        percentage: 33.3,
        features: { 満足度: 40, 生産性: 60 },
        insights: ['低い満足度と生産性']
      }
    ]
  };

  return (
    <div className="container max-w-7xl mx-auto px-4">
      <div className={`my-8 ${fadeInAnimation()}`}>
        <h1 className="text-3xl font-bold mb-6">データ分析</h1>

        <div className={`p-6 bg-card rounded-lg border shadow-sm mb-6 ${slideInAnimation("up", 100)}`}>
          <h2 className="text-xl font-semibold mb-2">分析ツール</h2>
          <p className="text-muted-foreground">
            さまざまな分析手法を用いて、企業のウェルネス状態を可視化・解析することができます。
            各タブを選択して、異なる分析手法やチャートを確認してください。
          </p>
        </div>
      </div>

      <Tabs defaultValue="monthly" className="w-full">
        <TabsList className="w-full justify-start mb-4 overflow-x-auto">
          <TabsTrigger value="monthly" className={zoomInAnimation(100)}>月次トレンド</TabsTrigger>
          <TabsTrigger value="timeseries" className={zoomInAnimation(150)}>時系列分析</TabsTrigger>
          <TabsTrigger value="survival" className={zoomInAnimation(200)}>生存分析</TabsTrigger>
          <TabsTrigger value="sentiment" className={zoomInAnimation(250)}>感情分析</TabsTrigger>
          <TabsTrigger value="textmining" className={zoomInAnimation(300)}>テキストマイニング</TabsTrigger>
          <TabsTrigger value="cluster" className={zoomInAnimation(350)}>クラスター分析</TabsTrigger>
          <TabsTrigger value="bayesian" className={zoomInAnimation(400)}>ベイジアン分析</TabsTrigger>
        </TabsList>

        <TabsContent value="monthly" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">月次トレンド分析</h2>
            <MonthlyTrendChart data={monthlyData} />
          </Card>
        </TabsContent>

        <TabsContent value="timeseries" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">時系列分析</h2>
            <div className="h-[400px] w-full">
              <TimeSeriesChart data={timeSeriesData} />
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="survival" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">生存分析</h2>
            <div className="h-[400px] w-full">
              <SurvivalCurveChart curves={survivalCurves} />
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="sentiment" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">感情分析</h2>
            <div className="flex justify-center">
              <SentimentGauge positive={65} negative={15} neutral={20} size={300} />
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="textmining" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">テキストマイニング</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className={slideInAnimation("right", 100)}>
                <h3 className="text-lg font-semibold mb-2">ワードクラウド</h3>
                <WordCloudChart data={wordCloudData} />
              </div>
              <div className={slideInAnimation("left", 100)}>
                <h3 className="text-lg font-semibold mb-2">トピック分析</h3>
                <TopicBarChart topics={topicData} />
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="cluster" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">クラスター分析</h2>
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
              <div className="md:col-span-8">
                <ClusterChart
                  data={clusterData.points}
                  clusters={clusterData.clusters}
                  selectedCluster={selectedCluster}
                  onClusterSelect={(id) => setSelectedCluster(id)}
                  xAxisLabel="従業員満足度"
                  yAxisLabel="生産性"
                />
              </div>
              <div className="md:col-span-4">
                <h3 className="text-lg font-semibold mb-2">相関マトリックス</h3>
                <CorrelationMatrix data={correlationData} />
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="bayesian" className={fadeInAnimation()}>
          <BayesianAnalysis companyId="demo-company-1" />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AnalysisPage;