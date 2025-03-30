import React, { useState } from 'react';
import { WellnessScoreChart } from '../components/dashboard/WellnessScoreChart';
import { WellnessMetrics } from '../components/dashboard/WellnessMetrics';
import { MetricCard } from '../components/dashboard/MetricCard';
import { DashboardCard } from '../components/dashboard/DashboardCard';
import { AnalysisResults } from '../components/dashboard/AnalysisResults';
import { ChevronDown, Settings, Filter, RefreshCw, ExternalLink, TrendingUp, Users, DollarSign, Activity } from 'lucide-react';

// モックデータ
const mockWellnessData = {
  labels: ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024'],
  datasets: [
    {
      label: 'ウェルネススコア',
      data: [65, 72, 78, 84, 88],
      borderColor: '#4f46e5',
      backgroundColor: 'rgba(79, 70, 229, 0.1)',
      tension: 0.3,
    },
    {
      label: '業界平均',
      data: [60, 62, 65, 67, 70],
      borderColor: '#94a3b8',
      backgroundColor: 'rgba(148, 163, 184, 0.1)',
      tension: 0.3,
      borderDash: [5, 5],
    }
  ]
};

const mockMetrics = [
  { name: '全体スコア', value: 85, change: 3.2, trend: 'up' },
  { name: 'メンタルヘルス', value: 82, change: 5.1, trend: 'up' },
  { name: 'ワークライフバランス', value: 78, change: 2.8, trend: 'up' },
  { name: '職場環境', value: 90, change: 1.5, trend: 'up' },
];

const mockCompanies = [
  { id: '1', name: 'テックスタート株式会社', score: 85, change: 3.2 },
  { id: '2', name: 'ヘルスケアイノベーション', score: 92, change: 5.5 },
  { id: '3', name: 'グリーンテック', score: 78, change: -1.2 },
  { id: '4', name: 'フューチャーデザイン', score: 81, change: 2.1 },
  { id: '5', name: 'ソフトイノベーションズ', score: 72, change: 4.8 },
];

const mockAnalysisResults = [
  {
    title: '従業員ウェルネスの最適化ポイント',
    description: 'メンタルヘルスサポートの強化とワークライフバランスの改善が最も効果的です',
    impact: 'high',
    metrics: ['エンゲージメント +15%', '離職率 -22%', '生産性 +8%']
  },
  {
    title: '投資リターン予測',
    description: 'ウェルネスプログラムへの投資により、3年間でROI 250%が期待できます',
    impact: 'medium',
    metrics: ['コスト削減 ¥1,200万', '生産性向上 ¥2,800万', '人材確保 ¥1,500万']
  },
  {
    title: '業界比較インサイト',
    description: 'あなたの企業は同業他社に比べてメンタルヘルス対策が15%優れています',
    impact: 'low',
    metrics: ['業界平均より +12pt', 'トップ25%に位置', '成長率 業界1位']
  }
];

const DashboardPage: React.FC = () => {
  const [timeRange, setTimeRange] = useState('3ヶ月');

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold">ダッシュボード</h1>
            <p className="text-gray-400 mt-1">ウェルネス分析の概要</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center bg-gray-800 rounded-lg p-2">
              <span className="text-sm text-gray-400 mr-2">期間:</span>
              <button className="text-sm flex items-center">
                {timeRange}
                <ChevronDown className="h-4 w-4 ml-1" />
              </button>
            </div>
            <button className="p-2 bg-gray-800 rounded-lg">
              <Settings className="h-5 w-5 text-gray-400" />
            </button>
            <button className="p-2 bg-gray-800 rounded-lg">
              <Filter className="h-5 w-5 text-gray-400" />
            </button>
            <button className="p-2 bg-gray-800 rounded-lg">
              <RefreshCw className="h-5 w-5 text-gray-400" />
            </button>
          </div>
        </div>

        {/* メインコンテンツ */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 左側：ウェルネススコアチャートとメトリクス */}
          <div className="lg:col-span-2 space-y-6">
            <DashboardCard title="ウェルネススコア推移" subtitle="直近3か月の変化：+8.9%">
              <div className="h-80">
                <WellnessScoreChart data={mockWellnessData} />
              </div>
            </DashboardCard>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {mockMetrics.map((metric, index) => (
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
                <AnalysisResults results={mockAnalysisResults} />
              </div>
            </DashboardCard>
          </div>

          {/* 右側：サマリーと企業リスト */}
          <div className="space-y-6">
            <DashboardCard title="企業概要" subtitle="登録企業のウェルネス状況">
              <WellnessMetrics />
            </DashboardCard>

            <DashboardCard title="主要指標" subtitle="前月比較">
              <div className="space-y-4 mt-4">
                <div className="flex justify-between items-center p-3 bg-gray-800 rounded-lg">
                  <div className="flex items-center">
                    <div className="bg-indigo-600 p-2 rounded mr-3">
                      <TrendingUp className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-400">平均ウェルネススコア</p>
                      <p className="text-lg font-bold">82.3</p>
                    </div>
                  </div>
                  <span className="text-green-500 text-sm font-medium">+3.5%</span>
                </div>

                <div className="flex justify-between items-center p-3 bg-gray-800 rounded-lg">
                  <div className="flex items-center">
                    <div className="bg-green-600 p-2 rounded mr-3">
                      <Users className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-400">従業員総数</p>
                      <p className="text-lg font-bold">1,245名</p>
                    </div>
                  </div>
                  <span className="text-green-500 text-sm font-medium">+12名</span>
                </div>

                <div className="flex justify-between items-center p-3 bg-gray-800 rounded-lg">
                  <div className="flex items-center">
                    <div className="bg-blue-600 p-2 rounded mr-3">
                      <DollarSign className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-400">ウェルネス投資ROI</p>
                      <p className="text-lg font-bold">167%</p>
                    </div>
                  </div>
                  <span className="text-green-500 text-sm font-medium">+22%</span>
                </div>

                <div className="flex justify-between items-center p-3 bg-gray-800 rounded-lg">
                  <div className="flex items-center">
                    <div className="bg-yellow-600 p-2 rounded mr-3">
                      <Activity className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-400">エンゲージメント率</p>
                      <p className="text-lg font-bold">78.5%</p>
                    </div>
                  </div>
                  <span className="text-green-500 text-sm font-medium">+5.2%</span>
                </div>
              </div>
            </DashboardCard>

            <DashboardCard title="トップ企業" subtitle="ウェルネススコア順">
              <div className="mt-2 space-y-2">
                {mockCompanies.map((company, index) => (
                  <div key={company.id} className="flex justify-between items-center p-3 bg-gray-800 rounded-lg hover:bg-gray-750">
                    <div className="flex items-center">
                      <div className="bg-gray-700 w-8 h-8 flex items-center justify-center rounded-full mr-3 font-medium">
                        {index + 1}
                      </div>
                      <p className="font-medium">{company.name}</p>
                    </div>
                    <div className="text-right">
                      <p className={`font-bold ${company.score >= 80 ? 'text-green-500' : company.score >= 70 ? 'text-blue-500' : 'text-yellow-500'}`}>
                        {company.score}
                      </p>
                      <p className={`text-xs ${company.change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {company.change >= 0 ? '+' : ''}{company.change}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </DashboardCard>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;