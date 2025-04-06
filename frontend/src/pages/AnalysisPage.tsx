import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { TimeSeriesChart, TimeSeriesData as ChartTimeSeriesData } from '@/components/charts/TimeSeriesChart';
import { SentimentGauge } from '@/components/charts/SentimentGauge';
import { SurvivalCurveChart } from '@/components/charts/SurvivalCurveChart';
import type { SurvivalCurve } from '@/components/charts/SurvivalCurveChart';
import { WordCloudChart } from '@/components/charts/WordCloudChart';
import { TopicBarChart } from '@/components/charts/TopicBarChart';
import { ClusterChart } from '@/components/charts/ClusterChart';
import { CorrelationMatrix } from '@/components/charts/CorrelationMatrix';
import { BayesianAnalysis } from '@/components/analysis/BayesianAnalysis';
import { useWellnessAnalysis, CorrelationData as HookCorrelationData, TimeSeriesData as HookTimeSeriesData, SurvivalAnalysisData } from '@/hooks/useWellnessAnalysis';
import { useTextMining, KeywordAnalysis, TopicAnalysis } from '@/hooks/useTextMining';
import { Loader2 } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { fadeInAnimation, slideInAnimation, zoomInAnimation } from '@/lib/utils';

const AnalysisPage: React.FC = () => {
  const [selectedCluster, setSelectedCluster] = useState<string | null>(null);

  const {
    correlationData: wellnessCorrelationData,
    clusterData,
    timeSeriesData: wellnessTimeSeriesData,
    survivalData,
    pcaData,
    descriptiveStats,
    loading: wellnessLoading,
    error: wellnessError,
    refreshData: refreshWellnessData,
    applyFilters: applyWellnessFilters
  } = useWellnessAnalysis();

  const {
    sentiment,
    keywords,
    phrases,
    entities,
    topics,
    insights: textInsights,
    loading: textLoading,
    error: textError,
    refreshData: refreshTextData,
    applyFilters: applyTextFilters
  } = useTextMining();

  const loading = wellnessLoading || textLoading;
  const error = wellnessError || textError;

  const formatTimeSeriesDataForChart = (data: HookTimeSeriesData | undefined): ChartTimeSeriesData | undefined => {
    if (!data || !data.labels || !data.datasets) return undefined;
    return {
      dates: data.labels,
      series: data.datasets.map((ds) => ({
        name: ds.label,
        data: ds.data,
        color: ds.borderColor,
        isMainMetric: ds.label.includes('スコア'),
        scale: 1
      })),
      annotations: [],
      insights: data.insights || []
    };
  };
  const formattedTimeSeriesData = formatTimeSeriesDataForChart(wellnessTimeSeriesData);

  const formatSurvivalDataForChart = (data: SurvivalAnalysisData | undefined): SurvivalCurve[] | undefined => {
    if (!data || !data.segments) return undefined;
    return data.segments.map((seg) => ({
      name: seg.name,
      color: seg.color,
      data: seg.data.map(point => ({ time: point.time, probability: point.survival }))
    }));
  };
  const formattedSurvivalCurves = formatSurvivalDataForChart(survivalData);

  const formatCorrelationDataForChart = (data: HookCorrelationData | undefined): { x: string; y: string; value: number; }[] | undefined => {
    if (!data || !data.matrix || !data.variables) return undefined;
    const points: { x: string; y: string; value: number; }[] = [];
    for (let i = 0; i < data.variables.length; i++) {
      for (let j = 0; j < data.variables.length; j++) {
        points.push({
          x: data.variables[i],
          y: data.variables[j],
          value: data.matrix[i][j]
        });
      }
    }
    return points;
  };
  const formattedCorrelationPoints = formatCorrelationDataForChart(wellnessCorrelationData);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <Loader2 className="h-16 w-16 animate-spin text-primary" />
        <p className="ml-4 text-lg">分析データを読み込んでいます...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container max-w-7xl mx-auto px-4 py-8">
        <Alert variant="destructive">
          <AlertTitle>エラー</AlertTitle>
          <AlertDescription>
            データの読み込みに失敗しました: {error.message}
            <button onClick={() => { refreshWellnessData(); refreshTextData(); }} className="ml-4 px-2 py-1 bg-red-700 hover:bg-red-800 rounded text-sm text-white">再試行</button>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

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

      <Tabs defaultValue="timeseries" className="w-full">
        <TabsList className="w-full justify-start mb-4 overflow-x-auto">
          <TabsTrigger value="timeseries" className={zoomInAnimation(150)}>時系列分析</TabsTrigger>
          <TabsTrigger value="survival" className={zoomInAnimation(200)}>生存分析</TabsTrigger>
          <TabsTrigger value="sentiment" className={zoomInAnimation(250)}>感情分析</TabsTrigger>
          <TabsTrigger value="textmining" className={zoomInAnimation(300)}>テキストマイニング</TabsTrigger>
          <TabsTrigger value="cluster" className={zoomInAnimation(350)}>クラスター分析</TabsTrigger>
          <TabsTrigger value="correlation" className={zoomInAnimation(400)}>相関分析</TabsTrigger>
        </TabsList>

        <TabsContent value="timeseries" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">時系列分析</h2>
            <div className="h-[400px] w-full">
              {formattedTimeSeriesData ? (
                <TimeSeriesChart data={formattedTimeSeriesData} />
              ) : (
                <p className="text-muted-foreground">時系列データがありません。</p>
              )}
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="survival" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">生存分析</h2>
            <div className="h-[400px] w-full">
              {formattedSurvivalCurves ? (
                <SurvivalCurveChart curves={formattedSurvivalCurves} />
              ) : (
                <p className="text-muted-foreground">生存分析データがありません。</p>
              )}
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="sentiment" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">感情分析</h2>
            <div className="flex justify-center">
              {sentiment ? (
                <SentimentGauge
                  positive={Math.round(sentiment.positive * 100)}
                  negative={Math.round(sentiment.negative * 100)}
                  neutral={Math.round(sentiment.neutral * 100)}
                  size={300}
                />
              ) : (
                <p className="text-muted-foreground">感情分析データがありません。</p>
              )}
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="textmining" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">テキストマイニング</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className={slideInAnimation("right", 100)}>
                <h3 className="text-lg font-semibold mb-2">ワードクラウド</h3>
                {keywords && keywords.length > 0 ? (
                  <WordCloudChart data={keywords.map((k: KeywordAnalysis) => ({ text: k.word, value: k.count }))} />
                ) : (
                  <p className="text-muted-foreground">キーワードデータがありません。</p>
                )}
              </div>
              <div className={slideInAnimation("left", 100)}>
                <h3 className="text-lg font-semibold mb-2">トピック分析</h3>
                {topics && topics.length > 0 ? (
                  <TopicBarChart topics={topics.map((t: TopicAnalysis) => ({ ...t, sentiment: Math.random() }))} />
                ) : (
                  <p className="text-muted-foreground">トピックデータがありません。</p>
                )}
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="cluster" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">クラスター分析</h2>
            {clusterData ? (
              <ClusterChart
                data={clusterData.points}
                clusters={clusterData.clusters}
                selectedCluster={selectedCluster}
                onClusterSelect={(id) => setSelectedCluster(id)}
                xAxisLabel={clusterData.xAxisLabel}
                yAxisLabel={clusterData.yAxisLabel}
              />
            ) : (
              <p className="text-muted-foreground">クラスター分析データがありません。</p>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="correlation" className={fadeInAnimation()}>
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">相関分析</h2>
            {formattedCorrelationPoints ? (
              <CorrelationMatrix data={formattedCorrelationPoints} />
            ) : (
              <p className="text-muted-foreground">相関分析データがありません。</p>
            )}
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AnalysisPage;