import React from 'react';
import { Link } from 'react-router-dom';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardFooter
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import {
  LayoutDashboard,
  Building,
  TrendingUp,
  FileText,
  ArrowRight
} from 'lucide-react';
import { Company } from '@/types/company';
// recharts import
import {
  LineChart as ReLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart as RePieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis
} from 'recharts';

// モックデータ
const wellnessScoreData = [
  { month: '1月', スコア: 78 },
  { month: '2月', スコア: 82 },
  { month: '3月', スコア: 80 },
  { month: '4月', スコア: 85 },
  { month: '5月', スコア: 87 },
  { month: '6月', スコア: 84 },
];

const recentCompanies: Partial<Company>[] = [
  { id: '1', name: 'テックスタート株式会社', industry: 'SaaS', wellnessScore: 85 },
  { id: '2', name: 'ヘルスケアイノベーション', industry: 'ヘルスケア', wellnessScore: 92 },
  { id: '3', name: 'グリーンテック', industry: 'クリーンテック', wellnessScore: 78 },
];

const categoryData = [
  { name: 'メンタルヘルス', value: 85 },
  { name: 'フィジカルヘルス', value: 72 },
  { name: 'ワークライフバランス', value: 78 },
  { name: '職場環境', value: 88 },
  { name: '福利厚生', value: 81 },
];

const scatterData: Array<Partial<Company> & { size: number }> = [
  { name: 'テックスタート', wellnessScore: 85, growthRate: 45, industry: 'SaaS', size: 45 },
  { name: 'ヘルスケアイノベーション', wellnessScore: 92, growthRate: 65, industry: 'ヘルスケア', size: 70 },
  { name: 'グリーンテック', wellnessScore: 78, growthRate: 30, industry: 'クリーンテック', size: 38 },
  { name: 'フィンテックラボ', wellnessScore: 83, growthRate: 50, industry: 'フィンテック', size: 25 },
  { name: 'AIリサーチ', wellnessScore: 87, growthRate: 55, industry: 'AI', size: 30 },
  { name: 'モビリティフューチャー', wellnessScore: 75, growthRate: 35, industry: 'モビリティ', size: 55 },
];

// パイチャートの色
const COLORS = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

const HomePage = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="bg-gray-800 rounded-lg shadow-lg p-6">
        <h1 className="text-2xl font-bold mb-4 text-white">スタートアップウェルネス分析ダッシュボード</h1>
        <p className="text-gray-300 mb-6">
          スタートアップの健全性を分析し、成長を促進するためのツールです。
          さまざまな分析機能を使って、あなたのスタートアップの状態を把握しましょう。
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gray-700 rounded-lg p-4 shadow border border-blue-900">
            <h2 className="text-lg font-semibold text-blue-300 mb-2">企業管理</h2>
            <p className="text-gray-300 mb-4">スタートアップの基本情報を登録・管理できます。</p>
            <Link
              to="/startups/add"
              className="inline-block bg-blue-600 text-white rounded px-4 py-2 hover:bg-blue-700"
            >
              企業を追加する
            </Link>
          </div>

          <div className="bg-gray-700 rounded-lg p-4 shadow border border-purple-900">
            <h2 className="text-lg font-semibold text-purple-300 mb-2">ウェルネス分析</h2>
            <p className="text-gray-300 mb-4">
              登録した企業の健全性スコアを確認できます。
            </p>
            <Link
              to="/analysis"
              className="inline-block bg-purple-600 text-white rounded px-4 py-2 hover:bg-purple-700"
            >
              分析を開始する
            </Link>
          </div>

          <div className="bg-gray-700 rounded-lg p-4 shadow border border-green-900">
            <h2 className="text-lg font-semibold text-green-300 mb-2">レポート</h2>
            <p className="text-gray-300 mb-4">
              分析結果をもとにしたレポートを生成できます。
            </p>
            <Link
              to="/reports"
              className="inline-block bg-green-600 text-white rounded px-4 py-2 hover:bg-green-700"
            >
              レポートを作成する
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;