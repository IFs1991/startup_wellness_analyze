import React, { useState } from 'react';
import { useDashboardData } from '@/hooks/useDashboardData';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useChartTheme } from '@/hooks/useChartTheme';
import { WellnessMetrics } from '../components/dashboard/WellnessMetrics';
import { MetricCard } from '../components/dashboard/MetricCard';
import { DashboardCard } from '../components/dashboard/DashboardCard';
import { ChevronDown, Settings, Filter, RefreshCw, ExternalLink, TrendingUp, Users, DollarSign, Activity, Loader2 } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const DashboardPage: React.FC = () => {
  const [timeRange, setTimeRange] = useState('3m');
  const theme = useChartTheme();

  const {
    wellnessScores,
    metrics,
    recentActivities,
    topCompanies,
    analysisInsights,
    loading,
    error,
    refreshData,
    getDataForPeriod
  } = useDashboardData();

  const handleTimeRangeChange = (value: string) => {
    setTimeRange(value);
    getDataForPeriod(value);
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen bg-gray-900 text-white">
        <Loader2 className="h-16 w-16 animate-spin text-indigo-500" />
        <p className="ml-4 text-lg">データを読み込んでいます...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 text-white p-8">
        <Alert variant="destructive">
          <AlertTitle>エラー</AlertTitle>
          <AlertDescription>
            ダッシュボードデータの読み込みに失敗しました: {error.message}
            <button onClick={refreshData} className="ml-4 px-2 py-1 bg-red-700 hover:bg-red-800 rounded text-sm">再試行</button>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const formattedMetrics = [
    { name: '全体スコア', value: metrics?.averageScore ?? 0, change: metrics?.scoreChange ?? 0, trend: (metrics?.scoreChange ?? 0) >= 0 ? 'up' : 'down' },
    { name: 'トップ企業数', value: metrics?.topPerformers ?? 0, change: 0, trend: 'neutral' },
    { name: 'エンゲージメント', value: metrics?.engagementRate ?? 0, change: 0, trend: 'neutral' },
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold">ダッシュボード</h1>
            <p className="text-gray-400 mt-1">ウェルネス分析の概要</p>
          </div>
          <div className="flex items-center gap-3">
            <Select value={timeRange} onValueChange={handleTimeRangeChange}>
              <SelectTrigger className="w-[180px] bg-gray-800 border-gray-700">
                <SelectValue placeholder="期間を選択" />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 text-white border-gray-700">
                <SelectItem value="1m">過去1ヶ月</SelectItem>
                <SelectItem value="3m">過去3ヶ月</SelectItem>
                <SelectItem value="6m">過去6ヶ月</SelectItem>
                <SelectItem value="1y">過去1年</SelectItem>
                <SelectItem value="all">全期間</SelectItem>
              </SelectContent>
            </Select>
            <button className="p-2 bg-gray-800 rounded-lg">
              <Settings className="h-5 w-5 text-gray-400" />
            </button>
            <button className="p-2 bg-gray-800 rounded-lg">
              <Filter className="h-5 w-5 text-gray-400" />
            </button>
            <button onClick={refreshData} className="p-2 bg-gray-800 rounded-lg">
              <RefreshCw className="h-5 w-5 text-gray-400" />
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <DashboardCard title="ウェルネススコア推移" subtitle={`直近${timeRange}の変化：${metrics?.scoreChange ?? 0 >= 0 ? '+' : ''}${metrics?.scoreChange?.toFixed(1) ?? 'N/A'}%`}>
              <div className="h-80">
                {wellnessScores && wellnessScores.labels && wellnessScores.datasets ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={wellnessScores.labels.map((label, index) => ({
                        name: label,
                        ...(wellnessScores.datasets.reduce((acc, ds) => {
                            acc[ds.label] = ds.data[index];
                            return acc;
                        }, {} as Record<string, number>))
                      }))}
                       margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={theme.grid.stroke} />
                      <XAxis dataKey="name" stroke={theme.axis.stroke} fontSize={theme.axis.fontSize} />
                      <YAxis stroke={theme.axis.stroke} fontSize={theme.axis.fontSize} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: theme.tooltip.background,
                          border: `1px solid ${theme.tooltip.border}`,
                          borderRadius: '8px',
                          color: theme.tooltip.color
                        }}
                      />
                      <Legend wrapperStyle={{ color: theme.axis.stroke }} />
                      {wellnessScores.datasets.map((ds) => (
                        <Line
                          key={ds.label}
                          type="monotone"
                          dataKey={ds.label}
                          stroke={ds.borderColor}
                          strokeWidth={2}
                          dot={false}
                          activeDot={{ r: 8 }}
                          strokeDasharray={ds.borderDash?.join(' ')}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                ) : <p>スコアデータがありません。</p>}
              </div>
            </DashboardCard>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {formattedMetrics.map((metric, index) => (
                <MetricCard
                  key={index}
                  title={metric.name}
                  value={metric.value}
                  change={metric.change}
                  trend={metric.trend as 'up' | 'down' | 'neutral'}
                />
              ))}
            </div>

            <DashboardCard title="分析インサイト" actionText="全ての分析を見る" actionIcon={<ExternalLink className="h-4 w-4" />}>
              <div className="space-y-4">
                {analysisInsights && analysisInsights.length > 0 ? (
                  <ul className="space-y-3">
                    {analysisInsights.map((insight, index) => (
                      <li key={index} className="p-3 bg-gray-800 rounded-lg">
                        <h4 className="font-semibold text-indigo-400">{insight.title}</h4>
                        <p className="text-sm text-gray-300 mt-1">{insight.description}</p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          {insight.metrics.map((metric, mIndex) => (
                            <span key={mIndex} className="px-2 py-1 bg-gray-700 text-xs rounded">
                              {metric}
                            </span>
                          ))}
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p>分析インサイトはありません。</p>
                )}
              </div>
            </DashboardCard>
          </div>

          <div className="space-y-6">
            <DashboardCard title="企業概要" subtitle="登録企業のウェルネス状況">
              {metrics ? (
                <WellnessMetrics
                  averageScore={metrics.averageScore}
                  scoreChange={metrics.scoreChange}
                  topPerformers={metrics.topPerformers}
                  engagementRate={metrics.engagementRate}
                />
              ) : (
                <p>企業概要データがありません。</p>
              )}
            </DashboardCard>

            <DashboardCard title="主要指標" subtitle="前期間比較">
              <div className="space-y-4 mt-4">
                <div className="flex justify-between items-center p-3 bg-gray-800 rounded-lg">
                  <div className="flex items-center">
                    <div className="bg-indigo-600 p-2 rounded mr-3">
                      <TrendingUp className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-400">平均ウェルネススコア</p>
                      <p className="text-lg font-bold">{metrics?.averageScore?.toFixed(1) ?? 'N/A'}</p>
                    </div>
                  </div>
                  <span className={`${(metrics?.scoreChange ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'} text-sm font-medium`}>
                    {(metrics?.scoreChange ?? 0) >= 0 ? '+' : ''}{metrics?.scoreChange?.toFixed(1) ?? 'N/A'}%
                  </span>
                </div>

                <div className="flex justify-between items-center p-3 bg-gray-800 rounded-lg">
                  <div className="flex items-center">
                    <div className="bg-green-600 p-2 rounded mr-3">
                      <Users className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-400">トップパフォーマー企業数</p>
                      <p className="text-lg font-bold">{metrics?.topPerformers ?? 'N/A'}社</p>
                    </div>
                  </div>
                </div>

                <div className="flex justify-between items-center p-3 bg-gray-800 rounded-lg">
                  <div className="flex items-center">
                    <div className="bg-yellow-600 p-2 rounded mr-3">
                      <Activity className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-400">エンゲージメント率</p>
                      <p className="text-lg font-bold">{metrics?.engagementRate?.toFixed(1) ?? 'N/A'}%</p>
                    </div>
                  </div>
                </div>
              </div>
            </DashboardCard>

            <DashboardCard title="トップ企業" subtitle="ウェルネススコア順">
              <div className="mt-2 space-y-2">
                {topCompanies && topCompanies.length > 0 ? (
                  topCompanies.map((company, index) => (
                    <div key={company.id} className="flex justify-between items-center p-3 bg-gray-800 rounded-lg hover:bg-gray-750">
                      <div className="flex items-center">
                        <div className="bg-gray-700 w-8 h-8 flex items-center justify-center rounded-full mr-3 font-medium">
                          {index + 1}
                        </div>
                        <p className="font-medium">{company.name}</p>
                      </div>
                      <div className="text-right">
                        <p className={`font-bold ${company.score >= 80 ? 'text-green-500' : company.score >= 70 ? 'text-blue-500' : 'text-yellow-500'}`}>
                          {company.score.toFixed(1)}
                        </p>
                        <p className={`text-xs ${company.change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                          {company.change >= 0 ? '+' : ''}{company.change.toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  ))
                ) : (
                  <p>企業データがありません。</p>
                )}
              </div>
            </DashboardCard>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;