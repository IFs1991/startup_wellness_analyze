import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  Pencil, MapPin, Users, Calendar, TrendingUp,
  AlertTriangle, Info, HelpCircle, DollarSign,
  ChevronLeft, BarChart2 as BarChart3,
  Building, Clock, Lightbulb, PieChart,
  ClipboardList, FileBarChart, FileSpreadsheet, Monitor,
  Mail, FileOutput, History, Download, Eye, Settings,
  File as FileText, CheckCircle, AlertCircle
} from 'lucide-react';
import { ExplanationPopup } from '@/components/ui/explanation-popup';
import { analysisExplanations } from '@/data/analysis-explanations';
import { EditCompanyDialog } from '@/components/companies/EditCompanyDialog';
import { StageInfo, CompanyDetail } from '@/types/company';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

// ステージ情報の定義
const stageInfoMap: Record<string, StageInfo> = {
  'pre-seed': { value: 'pre-seed', label: 'プレシード', color: 'bg-slate-100 text-slate-800' },
  'seed': { value: 'seed', label: 'シード', color: 'bg-emerald-100 text-emerald-800' },
  'early': { value: 'early', label: 'アーリー', color: 'bg-blue-100 text-blue-800' },
  'series-a': { value: 'series-a', label: 'シリーズA', color: 'bg-purple-100 text-purple-800' },
  'series-b': { value: 'series-b', label: 'シリーズB', color: 'bg-indigo-100 text-indigo-800' },
  'series-c': { value: 'series-c', label: 'シリーズC', color: 'bg-pink-100 text-pink-800' },
  'series-d': { value: 'series-d', label: 'シリーズD以降', color: 'bg-amber-100 text-amber-800' },
  'pre-ipo': { value: 'pre-ipo', label: 'プレIPO', color: 'bg-red-100 text-red-800' }
};

const CompanyDetailPage = () => {
  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);
  const [explanationContent, setExplanationContent] = useState<{
    title: string;
    description: string;
    businessValue: string;
    caution: string;
  } | null>(null);

  // ダイアログの開閉ハンドラー
  const handleOpenEditDialog = () => {
    setOpenEditDialog(true);
  };

  // 説明ポップアップの開閉
  const handleOpenExplanation = () => {
    // 説明コンテンツを設定
    setExplanationContent(analysisExplanations[analysisTab as keyof typeof analysisExplanations]);
    setShowExplanation(true);
  };

  const handleCloseExplanation = () => {
    setShowExplanation(false);
  };

  // スコアに基づいた色を取得する関数
  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 80) return 'text-green-500';
    if (score >= 70) return 'text-blue-500';
    if (score >= 60) return 'text-amber-500';
    return 'text-red-500';
  };

  // スコア表示のインジケーター
  const getScoreIndicator = (score: number) => {
    if (score >= 80) {
      return {
        icon: <CheckCircle className="h-4 w-4" />,
        label: '優良',
      };
    }
    if (score >= 60) {
      return {
        icon: <Info className="h-4 w-4" />,
        label: '良好',
      };
    }
    if (score >= 40) {
      return {
        icon: <AlertTriangle className="h-4 w-4" />,
        label: '要改善',
      };
    }
    return {
      icon: <AlertCircle className="h-4 w-4" />,
      label: '注意',
    };
  };

  // スコアに基づいた背景色を取得する関数
  const getScoreBgColor = (score: number) => {
    if (score >= 85) return 'bg-green-500';
    if (score >= 75) return 'bg-blue-500';
    if (score >= 65) return 'bg-amber-500';
    return 'bg-red-500';
  };

  // ダミーデータ（実際にはAPIから取得）
  const [company, setCompany] = useState<CompanyDetail>({
    id: "12345", // IDを固定値に設定
    name: "テックイノベート株式会社",
    industry: 'SaaS',
    wellnessScore: 82,
    foundedYear: 2020,
    employees: 45,
    location: '東京',
    revenue: '2.5億円',
    growthRate: '15%',
    ceo: '山田 太郎',
    stage: 'series-a',
    totalFunding: '5.2億円',
    fundingRounds: 3,
    description: 'クラウドベースの営業支援ツールを提供するB2B SaaS企業。月間売上の成長率は15%を維持し、顧客満足度も高い。',
    strengths: [
      '従業員満足度の高さ',
      'リモートワークのインフラ整備',
      '従業員のスキル開発制度'
    ],
    weaknesses: [
      'フィジカルヘルスの取り組み',
      '長時間労働の傾向'
    ],
    scoreBreakdown: [
      { category: 'メンタルヘルス', value: 85 },
      { category: 'フィジカルヘルス', value: 72 },
      { category: 'ワークライフバランス', value: 79 },
      { category: '職場環境', value: 88 },
      { category: '福利厚生', value: 81 },
      { category: 'キャリア開発', value: 90 }
    ],
    investments: [],
    score: 82,
    financials: {
      revenue: 250000000,
      growth: 15,
      profit: 50000000,
      history: []
    },
    wellness: {
      score: 82,
      engagement: 85,
      satisfaction: 87,
      workLife: 79,
      stress: 72,
      trends: []
    },
    surveys: []
  });

  // 会社情報の更新処理
  const handleCompanyUpdated = (updatedCompany: Partial<CompanyDetail>) => {
    setCompany({ ...company, ...updatedCompany });
  };

  // 相関分析データ
  const correlationData = [
    { item: 'ウェルネススコア', salesGrowth: 0.85, profitMargin: 0.72, employees: 0.45 },
    { item: 'エンゲージメント', salesGrowth: 0.78, profitMargin: 0.65, employees: 0.32 },
    { item: '満足度', salesGrowth: 0.82, profitMargin: 0.69, employees: 0.40 },
    { item: 'ワークライフバランス', salesGrowth: 0.71, profitMargin: 0.68, employees: 0.30 },
  ];

  // スコア表示のインジケーター
  const scoreIndicator = getScoreIndicator(company.wellnessScore ?? 0);

  // 現在選択されているタブを追跡するための状態
  const [currentTab, setCurrentTab] = useState<string>("overview");

  // 投資ラウンドのデータ（実際にはAPIから取得）
  const fundingRoundsData = [
    {
      id: "round-seed",
      date: "2020-05-15",
      amount: 50000000,
      round: "シード",
      investors: ["エンジェル投資家A", "シードファンドB"],
      postMoney: 200000000,
      wellnessScore: 75
    },
    {
      id: "round-series-a",
      date: "2021-08-22",
      amount: 300000000,
      round: "シリーズA",
      investors: ["ベンチャーキャピタルX", "コーポレートY"],
      postMoney: 1200000000,
      wellnessScore: 82
    },
    {
      id: "round-series-b",
      date: "2022-11-10",
      amount: 170000000,
      round: "シリーズB",
      investors: ["グロースファンドZ", "ストラテジックインベスターQ"],
      postMoney: 2500000000,
      wellnessScore: 87
    }
  ];

  // ROI予測データ
  const roiPredictionData = [
    { year: '2024', predicted: 15, average: 10 },
    { year: '2025', predicted: 22, average: 12 },
    { year: '2026', predicted: 30, average: 15 },
    { year: '2027', predicted: 45, average: 18 },
    { year: '2028', predicted: 55, average: 20 }
  ];

  // ウェルネススコアと資金調達の関係データ
  const wellnessFundingCorrelationData = [
    { quarter: 'Q1 2021', wellnessScore: 76, fundingSuccess: 35 },
    { quarter: 'Q2 2021', wellnessScore: 78, fundingSuccess: 40 },
    { quarter: 'Q3 2021', wellnessScore: 80, fundingSuccess: 48 },
    { quarter: 'Q4 2021', wellnessScore: 82, fundingSuccess: 55 },
    { quarter: 'Q1 2022', wellnessScore: 83, fundingSuccess: 62 },
    { quarter: 'Q2 2022', wellnessScore: 85, fundingSuccess: 68 },
    { quarter: 'Q3 2022', wellnessScore: 86, fundingSuccess: 72 },
    { quarter: 'Q4 2022', wellnessScore: 87, fundingSuccess: 75 },
  ];

  // 金額をフォーマットする関数
  const formatAmount = (amount: number): string => {
    if (amount >= 100000000) {
      return `${(amount / 100000000).toFixed(1)}億円`;
    } else if (amount >= 10000) {
      return `${(amount / 10000).toFixed(0)}万円`;
    }
    return `${amount.toLocaleString()}円`;
  };

  // 分析タブのサブタブ状態を追加
  const [analysisTab, setAnalysisTab] = useState<string>("correlation");

  return (
    <div className="container mx-auto p-4 max-w-full">
      <div className="mb-8">
        {/* 戻るボタンとアクション */}
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" asChild>
              <Link to="/companies">
                <ChevronLeft className="h-4 w-4 mr-1" />
            企業一覧に戻る
              </Link>
          </Button>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleOpenEditDialog}>
              <Pencil className="h-4 w-4 mr-1" />
              企業情報を編集
            </Button>
          </div>
        </div>

        {/* 会社概要ヘッダー */}
        <Card className="mb-6 relative overflow-visible">
          <div className={`absolute -top-3 right-5 ${getScoreBgColor(company.wellnessScore ?? 0)} text-white rounded-xl px-3 py-1 font-bold text-sm shadow-md z-10 flex items-center gap-1`}>
            {scoreIndicator.icon}
            <span>{company.wellnessScore ?? 0}点</span>
          </div>
          <CardContent className="p-6">
            <div className="flex items-start gap-4">
              <div>
                <Avatar className="h-16 w-16 border">
                  <AvatarFallback className="text-xl font-bold">
                    {company.name.charAt(0)}
                  </AvatarFallback>
                </Avatar>
              </div>
              <div className="flex-1">
                <div className="flex justify-between">
                  <div>
                    <h1 className="text-2xl font-bold mb-1">{company.name}</h1>
                    <div className="flex items-center gap-1 text-muted-foreground text-sm mb-2">
                      <Building className="h-4 w-4" />
                      <span>{company.industry}</span>
                      <span className="mx-1">•</span>
                      <Users className="h-4 w-4" />
                      <span>{company.employees}名</span>
                      <span className="mx-1">•</span>
                      <MapPin className="h-4 w-4" />
                      <span>{company.location}</span>
                      <span className="mx-1">•</span>
                      <Calendar className="h-4 w-4" />
                      <span>{company.foundedYear}年設立</span>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                      <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-200">
                        {company.industry}
                  </Badge>
                  <Badge className="bg-green-100 text-green-800 hover:bg-green-200 gap-1">
                    <TrendingUp className="h-3 w-3" />
                    成長率 +{company.growthRate}
                  </Badge>
                      <Badge variant="secondary" className={stageInfoMap[company.stage]?.color || 'bg-gray-100 text-gray-800'}>
                        {stageInfoMap[company.stage]?.label || company.stage}
                      </Badge>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* タブインターフェース */}
        <Tabs defaultValue="overview" value={currentTab} onValueChange={setCurrentTab} className="w-full">
          <TabsList className="mb-4">
            <TabsTrigger value="overview">
              <Building className="h-4 w-4 mr-2" />
              概要
            </TabsTrigger>
            <TabsTrigger value="analysis">
              <BarChart3 className="h-4 w-4 mr-2" />
              分析
            </TabsTrigger>
            <TabsTrigger value="funding">
              <DollarSign className="h-4 w-4 mr-2" />
              調達・投資分析
            </TabsTrigger>
            <TabsTrigger value="reports">
              <ClipboardList className="h-4 w-4 mr-2" />
              レポート
            </TabsTrigger>
          </TabsList>

          {/* 概要タブ */}
          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* 会社情報カード */}
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle className="text-lg">会社情報</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm mb-4">{company.description}</p>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-y-4">
                    <div>
                      <p className="text-sm text-muted-foreground">業種</p>
                      <p className="font-medium">{company.industry}</p>
                    </div>
              <div>
                <p className="text-sm text-muted-foreground">設立年</p>
                <p className="font-medium">{company.foundedYear}年</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">従業員数</p>
                <p className="font-medium">{company.employees}名</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">所在地</p>
                <p className="font-medium">{company.location}</p>
              </div>
                    <div>
                      <p className="text-sm text-muted-foreground">代表者</p>
                      <p className="font-medium">{company.ceo}</p>
                    </div>
              <div>
                <p className="text-sm text-muted-foreground">成長率</p>
                <p className="font-medium text-green-500">+{company.growthRate}</p>
                    </div>
              </div>
            </CardContent>
          </Card>

              {/* 調達情報サマリーカード */}
                <Card>
                  <CardHeader>
                  <CardTitle className="text-lg">資金調達</CardTitle>
                  </CardHeader>
                  <CardContent>
                  <div className="space-y-4">
                    <div>
                      <p className="text-sm text-muted-foreground">累計調達額</p>
                      <p className="text-2xl font-bold">{company.totalFunding}</p>
                      <p className="text-xs text-muted-foreground mt-1">ラウンド数: {fundingRoundsData.length}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">直近の調達</p>
                      <p className="font-medium">{fundingRoundsData[fundingRoundsData.length - 1]?.round}</p>
                      <p className="text-sm">{formatAmount(fundingRoundsData[fundingRoundsData.length - 1]?.amount)}</p>
                      <p className="text-xs text-muted-foreground">{new Date(fundingRoundsData[fundingRoundsData.length - 1]?.date).toLocaleDateString('ja-JP')}</p>
                    </div>
                    <div className="pt-2">
                      <Button variant="outline" size="sm" className="w-full" onClick={() => setCurrentTab("funding")}>
                        <DollarSign className="h-4 w-4 mr-1" />
                        詳細を見る
                      </Button>
                    </div>
                  </div>
                  </CardContent>
                </Card>
            </div>

            {/* 強み/弱みと詳細スコアの表示 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="text-lg font-semibold mb-2">ウェルネス評価</h3>
                <div className="grid grid-cols-1 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-green-500">強み</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ul className="list-disc pl-5 space-y-1">
                        {company.strengths?.map((strength: string, index: number) => (
                          <li key={index} className="text-sm">
                            {strength}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-red-500">改善点</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ul className="list-disc pl-5 space-y-1">
                        {company.weaknesses?.map((weakness: string, index: number) => (
                          <li key={index} className="text-sm">
                            {weakness}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                </div>
              </div>
              <div>
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle>スコア詳細</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {company.scoreBreakdown?.map((item: { category: string; value: number; description?: string }, index: number) => (
                      <div key={index}>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm">{item.category}</span>
                          <span className={`text-sm font-bold ${getScoreColor(item.value)}`}>
                            {item.value}/100
                          </span>
                        </div>
                        <Progress
                          value={item.value}
                          className={`h-2 ${
                            item.value >= 80
                              ? "bg-green-100"
                              : item.value >= 60
                              ? "bg-yellow-100"
                              : "bg-red-100"
                          }`}
                        />
                        {item.description && (
                          <p className="text-xs text-muted-foreground mt-1">
                            {item.description}
                          </p>
                        )}
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* 分析タブ */}
          <TabsContent value="analysis" className="space-y-4">
            {/* 分析サブタブ */}
            <Tabs defaultValue="correlation" value={analysisTab} onValueChange={setAnalysisTab}>
              <TabsList className="w-full mb-6 flex flex-wrap">
                <TabsTrigger value="correlation">相関分析</TabsTrigger>
                <TabsTrigger value="regression">回帰分析</TabsTrigger>
                <TabsTrigger value="bayesian">ベイズ推論</TabsTrigger>
                <TabsTrigger value="cluster">クラスター分析</TabsTrigger>
                <TabsTrigger value="time-series">時系列分析</TabsTrigger>
                <TabsTrigger value="relation">関連分析</TabsTrigger>
                <TabsTrigger value="text">テキスト分析</TabsTrigger>
                <TabsTrigger value="survival">生存分析</TabsTrigger>
              </TabsList>

              {/* 各分析タブのコンテンツ */}
              {["correlation", "regression", "bayesian", "cluster", "time-series", "relation", "text", "survival"].map((tab) => (
                <TabsContent key={tab} value={tab} className="space-y-6">
                  <Card>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle>{analysisExplanations[tab as keyof typeof analysisExplanations].title.split('（')[0]}</CardTitle>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="p-2"
                          onClick={handleOpenExplanation}
                        >
                          <HelpCircle className="h-5 w-5 text-blue-500" />
                        </Button>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {tab === "correlation"
                          ? "ウェルネススコアと財務指標の相関関係を分析します"
                          : `${analysisExplanations[tab as keyof typeof analysisExplanations].title.split('（')[0]}の結果がここに表示されます`}
                      </p>
                    </CardHeader>
                    <CardContent>
                      {tab === "correlation" ? (
                        <>
                          <h3 className="font-medium mb-3">相関係数行列</h3>
                          <div className="overflow-x-auto">
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
                          <div className="mt-6">
                            <h3 className="font-medium mb-3">グラフ表示</h3>
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
                          <p className="mt-4 text-sm">
                            分析結果：従業員のウェルネススコアと売上成長率の間に強い相関（0.85）が見られます。また、エンゲージメントと売上成長率の間にも強い相関（0.78）があります。これらの結果から、従業員のウェルビーイングへの投資が事業成果の向上につながる可能性が示唆されます。
                          </p>
                        </>
                      ) : (
                        <div className="p-8 text-center">
                          <p className="text-muted-foreground">
                            {analysisExplanations[tab as keyof typeof analysisExplanations].title.split('（')[0]}の結果を表示するには、データを選択してください。
                        </p>
                          <Button variant="outline" className="mt-4">
                            データ選択
                          </Button>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>
              ))}
            </Tabs>
          </TabsContent>

          {/* 調達・投資分析タブ */}
          <TabsContent value="funding" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* 調達ラウンドの詳細 */}
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <DollarSign className="h-5 w-5 mr-2" />
                    資金調達履歴
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="relative">
                    <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200"></div>
                    {fundingRoundsData.map((round, index) => (
                      <div key={round.id} className="relative pl-10 pb-8">
                        <div className="absolute left-3 top-1 h-6 w-6 rounded-full bg-primary flex items-center justify-center text-white font-bold text-xs">
                          {index + 1}
                        </div>
                        <div className="bg-gray-50 rounded-lg p-4">
                          <div className="flex flex-col md:flex-row md:justify-between md:items-center mb-3">
                            <div>
                              <h4 className="font-bold text-lg">{round.round}ラウンド</h4>
                              <p className="text-sm text-muted-foreground">
                                {new Date(round.date).toLocaleDateString('ja-JP')}
                              </p>
                            </div>
                            <div className="mt-2 md:mt-0">
                              <span className="font-bold text-xl text-primary">{formatAmount(round.amount)}</span>
                            </div>
                          </div>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <h5 className="text-sm font-semibold mb-1">投資家</h5>
                              <ul className="list-disc pl-5 text-sm">
                                {round.investors.map((investor, idx) => (
                                  <li key={idx}>{investor}</li>
                                ))}
                              </ul>
                            </div>
                            <div>
                              <div className="mb-2">
                                <h5 className="text-sm font-semibold mb-1">企業価値評価</h5>
                                <p className="font-medium">{formatAmount(round.postMoney)}</p>
                              </div>
                              <div>
                                <h5 className="text-sm font-semibold mb-1">当時のウェルネススコア</h5>
                                <div className="flex items-center">
                                  <Progress value={round.wellnessScore} className="h-2 w-24 mr-2" />
                                  <span className={`text-sm font-bold ${getScoreColor(round.wellnessScore)}`}>
                                    {round.wellnessScore}/100
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* 投資分析サマリー */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <TrendingUp className="h-5 w-5 mr-2" />
                    投資効果分析
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* ウェルネススコアと資金調達成功率の相関 */}
                  <div>
                    <h4 className="font-semibold mb-2 text-sm">ウェルネススコアと調達成功率</h4>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={wellnessFundingCorrelationData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="quarter" tick={{ fontSize: 10 }} />
                          <YAxis yAxisId="left" orientation="left" tick={{ fontSize: 10 }} />
                          <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10 }} />
                          <Tooltip />
                          <Legend wrapperStyle={{ fontSize: '10px' }} />
                          <Line yAxisId="left" type="monotone" dataKey="wellnessScore" name="ウェルネススコア" stroke="#8884d8" />
                          <Line yAxisId="right" type="monotone" dataKey="fundingSuccess" name="調達成功率(%)" stroke="#82ca9d" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* 推定ROI */}
                  <div>
                    <h4 className="font-semibold mb-2 text-sm flex items-center">
                      <Lightbulb className="h-4 w-4 mr-1 text-yellow-500" />
                      推定ROI予測
                    </h4>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={roiPredictionData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="year" tick={{ fontSize: 10 }} />
                          <YAxis tick={{ fontSize: 10 }} />
                          <Tooltip />
                          <Legend wrapperStyle={{ fontSize: '10px' }} />
                          <Bar dataKey="predicted" name="予測ROI(%)" fill="#8884d8" />
                          <Bar dataKey="average" name="業界平均ROI(%)" fill="#82ca9d" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="mt-4 p-3 bg-yellow-50 rounded-md border border-yellow-200">
                      <p className="text-xs">
                        <span className="font-semibold">分析結果:</span> ウェルネススコアの維持・向上により、
                        業界平均と比較して30%高いROIが期待されます。
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* 投資インサイト */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <PieChart className="h-5 w-5 mr-2" />
                  投資インサイト
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <h4 className="font-semibold mb-2 flex items-center text-blue-700">
                      <TrendingUp className="h-4 w-4 mr-1" />
                      成長指標
                    </h4>
                    <p className="text-sm">
                      ウェルネス指標の20%向上が資金調達成功率を35%上昇させる相関性が確認されています。
                      特に「職場環境」と「メンタルヘルス」の改善が投資家の意思決定に強い影響を与えています。
                    </p>
                  </div>
                  <div className="bg-green-50 p-4 rounded-lg border border-green-100">
                    <h4 className="font-semibold mb-2 flex items-center text-green-700">
                      <Users className="h-4 w-4 mr-1" />
                      人材活用
                    </h4>
                    <p className="text-sm">
                      従業員のエンゲージメント向上と離職率低下がシリーズAからシリーズBへの移行期間を平均3.5ヶ月短縮させる傾向にあります。
                      投資リターンの早期化に貢献しています。
                    </p>
                  </div>
                  <div className="bg-purple-50 p-4 rounded-lg border border-purple-100">
                    <h4 className="font-semibold mb-2 flex items-center text-purple-700">
                      <Clock className="h-4 w-4 mr-1" />
                      長期的価値
                    </h4>
                    <p className="text-sm">
                      ウェルネス投資の継続的実施企業は、5年後の企業価値が同業他社平均より42%高い結果となっています。
                      特に「キャリア開発」への投資が長期的ROIに強い相関を示しています。
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* レポートタブ（復元） */}
          <TabsContent value="reports" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <ClipboardList className="h-5 w-5 mr-2" />
                    レポート生成
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="mb-6">
                    <h3 className="text-sm font-semibold mb-2">レポート種類</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                      <Button variant="outline" className="justify-start">
                        <FileBarChart className="h-4 w-4 mr-2" />
                        財務パフォーマンス
                      </Button>
                      <Button variant="outline" className="justify-start">
                        <Users className="h-4 w-4 mr-2" />
                        従業員ウェルネス
                      </Button>
                      <Button variant="outline" className="justify-start">
                        <TrendingUp className="h-4 w-4 mr-2" />
                        成長分析
                      </Button>
                      <Button variant="outline" className="justify-start">
                        <BarChart3 className="h-4 w-4 mr-2" />
                        ベンチマーク比較
                      </Button>
                      <Button variant="outline" className="justify-start">
                        <FileText className="h-4 w-4 mr-2" />
                        総合評価
                      </Button>
                      <Button variant="outline" className="justify-start">
                        <Settings className="h-4 w-4 mr-2" />
                        カスタムレポート
                      </Button>
                    </div>
                  </div>

                  <div className="mb-6">
                    <h3 className="text-sm font-semibold mb-2">期間選択</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="text-sm text-muted-foreground">開始日</label>
                        <input type="date" className="w-full mt-1 p-2 rounded border" />
                      </div>
                      <div>
                        <label className="text-sm text-muted-foreground">終了日</label>
                        <input type="date" className="w-full mt-1 p-2 rounded border" />
                      </div>
                    </div>
                  </div>

                  <div className="mb-6">
                    <h3 className="text-sm font-semibold mb-2">出力形式</h3>
                    <div className="flex flex-wrap gap-3">
                      <Button variant="secondary" size="sm">
                        <FileText className="h-4 w-4 mr-1" />
                        PDF
                      </Button>
                      <Button variant="outline" size="sm">
                        <FileSpreadsheet className="h-4 w-4 mr-1" />
                        Excel
                      </Button>
                      <Button variant="outline" size="sm">
                        <Monitor className="h-4 w-4 mr-1" />
                        ダッシュボード
                      </Button>
                      <Button variant="outline" size="sm">
                        <Mail className="h-4 w-4 mr-1" />
                        メール
                      </Button>
                    </div>
                  </div>

                  <Button className="mt-4">
                    <FileOutput className="h-4 w-4 mr-2" />
                    レポートを生成
                  </Button>
                </CardContent>
              </Card>

            <Card>
              <CardHeader>
                  <CardTitle className="flex items-center">
                    <History className="h-5 w-5 mr-2" />
                    過去のレポート
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="border rounded-md p-3">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-medium">四半期業績レポート</p>
                          <p className="text-xs text-muted-foreground">2023年Q3（7月-9月）</p>
                        </div>
                        <Badge>PDF</Badge>
                      </div>
                      <div className="flex gap-2 mt-2">
                        <Button variant="outline" size="sm">
                          <Download className="h-3 w-3 mr-1" />
                          ダウンロード
                        </Button>
                        <Button variant="ghost" size="sm">
                          <Eye className="h-3 w-3 mr-1" />
                          プレビュー
                        </Button>
                      </div>
                    </div>

                    <div className="border rounded-md p-3">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-medium">ウェルネス分析</p>
                          <p className="text-xs text-muted-foreground">2023年8月</p>
                        </div>
                        <Badge>Excel</Badge>
                      </div>
                      <div className="flex gap-2 mt-2">
                        <Button variant="outline" size="sm">
                          <Download className="h-3 w-3 mr-1" />
                          ダウンロード
                        </Button>
                        <Button variant="ghost" size="sm">
                          <Eye className="h-3 w-3 mr-1" />
                          プレビュー
                        </Button>
                      </div>
                    </div>

                    <div className="border rounded-md p-3">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-medium">成長予測レポート</p>
                          <p className="text-xs text-muted-foreground">2023年7月</p>
                        </div>
                        <Badge>PDF</Badge>
                      </div>
                      <div className="flex gap-2 mt-2">
                        <Button variant="outline" size="sm">
                          <Download className="h-3 w-3 mr-1" />
                          ダウンロード
                        </Button>
                        <Button variant="ghost" size="sm">
                          <Eye className="h-3 w-3 mr-1" />
                          プレビュー
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* ExplanationPopupを実際に使用 */}
      {explanationContent && (
        <ExplanationPopup
          isOpen={showExplanation}
          onClose={handleCloseExplanation}
          content={explanationContent}
        />
      )}

      {/* 会社情報編集ダイアログ */}
      <EditCompanyDialog
        company={company}
        isOpen={openEditDialog}
        onOpenChange={setOpenEditDialog}
        onCompanyUpdated={handleCompanyUpdated}
      />
    </div>
  );
};

export default CompanyDetailPage;