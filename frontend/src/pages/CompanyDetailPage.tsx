import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, Building, Users, MapPin, Calendar, TrendingUp, Award } from 'lucide-react';
import {
  Box,
  Typography,
  Button,
  Tabs,
  Tab,
  Card,
  CardContent,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import EditIcon from '@mui/icons-material/Edit';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {
  CorrelationMatrix,
  BayesianChart,
  ClusterChart,
  TimeSeriesChart,
  RegressionAnalysisChart,
  SentimentGauge,
  AssociationRulesChart,
  TextAnalysisChart,
  SurvivalCurveChart
} from '../components/charts';

// モック企業データの型定義
interface Company {
  id: string;
  name: string;
  industry: string;
  stage: string;
  location: string;
  employeesCount: number;
  wellnessScore: number;
  growthRate: number;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`company-tabpanel-${index}`}
      aria-labelledby={`company-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `company-tab-${index}`,
    'aria-controls': `company-tabpanel-${index}`,
  };
}

const CompanyDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);
  const [analysisTabValue, setAnalysisTabValue] = useState(0);
  const [company, setCompany] = useState<Company | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 実際の実装ではAPIから企業データを取得しますが、
    // ここではモックデータをシミュレートします
    const mockCompanies: Company[] = [
      {
        id: '1',
        name: 'テックスタート株式会社',
        industry: 'SaaS',
        employeesCount: 45,
        location: '東京',
        stage: 'シリーズA',
        wellnessScore: 85,
        growthRate: 15
      },
      {
        id: '2',
        name: 'ヘルスケアイノベーション',
        industry: 'ヘルスケア',
        employeesCount: 120,
        location: '大阪',
        stage: 'シリーズB',
        wellnessScore: 92,
        growthRate: 25
      },
      {
        id: '3',
        name: 'グリーンテック',
        industry: 'クリーンテック',
        employeesCount: 15,
        location: '福岡',
        stage: 'シード',
        wellnessScore: 78,
        growthRate: 8
      },
      {
        id: '4',
        name: 'フューチャーデザイン',
        industry: 'デザイン',
        employeesCount: 28,
        location: '京都',
        stage: 'シリーズA',
        wellnessScore: 81,
        growthRate: 12
      },
      {
        id: '5',
        name: 'ソフトイノベーションズ',
        industry: 'ソフトウェア',
        employeesCount: 52,
        location: '名古屋',
        stage: 'シリーズB',
        wellnessScore: 76,
        growthRate: 18
      },
    ];

    // IDで企業を検索
    const foundCompany = mockCompanies.find(c => c.id === id);

    // データ取得をシミュレート
    setTimeout(() => {
      setCompany(foundCompany || null);
      setLoading(false);
    }, 500);
  }, [id]);

  // スコアに基づいた色を取得する関数
  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-500';
    if (score >= 80) return 'text-blue-500';
    if (score >= 70) return 'text-amber-500';
    return 'text-red-500';
  };

  // 成長率の色を取得する関数
  const getGrowthColor = (rate: number) => {
    if (rate >= 20) return 'text-green-500';
    if (rate >= 10) return 'text-blue-500';
    if (rate >= 5) return 'text-amber-500';
    return 'text-gray-400';
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleAnalysisTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setAnalysisTabValue(newValue);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent align-[-0.125em] text-blue-500 motion-reduce:animate-[spin_1.5s_linear_infinite]"></div>
          <p className="mt-4">データを読み込んでいます...</p>
        </div>
      </div>
    );
  }

  if (!company) {
    return (
      <div className="min-h-screen bg-gray-900 text-white p-6">
        <div className="max-w-4xl mx-auto">
          <Link to="/companies" className="inline-flex items-center text-blue-500 hover:text-blue-400 mb-6">
            <ArrowLeft className="mr-2 h-4 w-4" />
            企業一覧に戻る
          </Link>

          <div className="bg-gray-800 rounded-lg p-8 text-center">
            <h1 className="text-2xl font-bold mb-4">企業が見つかりませんでした</h1>
            <p className="text-gray-400 mb-6">指定されたIDの企業情報は存在しません。</p>
            <Link to="/companies" className="inline-flex items-center bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition duration-150">
              企業一覧に戻る
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/companies')}
          variant="outlined"
        >
          戻る
        </Button>
        <Button
          startIcon={<EditIcon />}
          variant="contained"
          color="primary"
        >
          編集
        </Button>
      </Box>

      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1" sx={{ flexGrow: 1 }}>
          {company.name}
        </Typography>
        <Chip label={company.industry} color="primary" sx={{ mr: 1 }} />
      </Box>

      <Typography variant="body1" sx={{ mb: 3 }}>
        {company.name}の企業ウェルネススコアは{company.wellnessScore}点で、
        {company.wellnessScore >= 90 ? '非常に高い水準にあります。従業員満足度や組織の健全性において模範的な企業です。' :
         company.wellnessScore >= 80 ? '良好な水準にあります。強みを活かし、さらなる改善の余地もあります。' :
         company.wellnessScore >= 70 ? '平均的な水準にあります。いくつかの重要な改善点に取り組むことをお勧めします。' :
         '改善が必要な状況です。組織文化や従業員満足度の向上に焦点を当てることをお勧めします。'}
      </Typography>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary">
                従業員数
              </Typography>
              <Typography variant="h6">
                {company.employeesCount}名
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary">
                所在地
              </Typography>
              <Typography variant="h6">
                {company.location}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary">
                成長率
              </Typography>
              <Typography variant="h6" color="success.main">
                +{company.growthRate}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary">
                ステータス
              </Typography>
              <Typography variant="h6">
                {company.wellnessScore >= 90 ? '優秀' :
                 company.wellnessScore >= 80 ? '良好' :
                 company.wellnessScore >= 70 ? '平均的' : '要改善'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="company tabs">
            <Tab label="概要" {...a11yProps(0)} />
            <Tab label="分析" {...a11yProps(1)} />
            <Tab label="レポート" {...a11yProps(2)} />
            <Tab label="データ管理" {...a11yProps(3)} />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    会社概要
                  </Typography>
                  <Typography variant="body2" paragraph>
                    <strong>業界:</strong> {company.industry}
                  </Typography>
                  <Typography variant="body2" paragraph>
                    <strong>従業員数:</strong> {company.employeesCount}名
                  </Typography>
                  <Typography variant="body2" paragraph>
                    <strong>所在地:</strong> {company.location}
                  </Typography>
                  <Typography variant="body2" paragraph>
                    <strong>成長率:</strong> {company.growthRate}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    ウェルネススコア
                  </Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                    <SentimentGauge value={company.wellnessScore} maxValue={100} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
            <Tabs
              value={analysisTabValue}
              onChange={handleAnalysisTabChange}
              aria-label="analysis tabs"
              variant="scrollable"
              scrollButtons="auto"
            >
              <Tab label="相関分析" />
              <Tab label="回帰分析" />
              <Tab label="ベイズ推論" />
              <Tab label="クラスター分析" />
              <Tab label="時系列分析" />
              <Tab label="関連分析" />
              <Tab label="テキスト分析" />
              <Tab label="生存分析" />
            </Tabs>
          </Box>

          {analysisTabValue === 0 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                相関分析
              </Typography>
              <Typography variant="body2" paragraph>
                ウェルネススコアと財務指標の相関関係を分析します
              </Typography>

              <Accordion defaultExpanded>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="correlation-content"
                  id="correlation-header"
                >
                  <Typography>相関分析とは？</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography>
                    相関分析は「2つの指標の間に関連性があるか」を測定する手法です。例えば、従業員の満足度と会社の売上の間に関連性があるかを-1から10数値で示します。+1に近いほど強い正の相関（一方が上がると他方も上がる）、-1に近いほど強い負の相関（一方が上がると他方は下がる）を表します。
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    ビジネス価値：従業員のエンゲージメントと顧客満足度の間に強い相関（0.78）があるという発見は、従業員満足度向上への投資が最終的に向上につながる可能性を示します。これにより、限られたリソースを最も効果的な領域に集中できます。
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    注意点：相関関係は因果関係を必ずしも意味しません。相関が見つかったら、その関係をより深く調査することが重要です。
                  </Typography>
                </AccordionDetails>
              </Accordion>

              <Box sx={{ mt: 2 }}>
                <Typography variant="h6" gutterBottom>
                  相関係数行列
                </Typography>
                <CorrelationMatrix
                  data={[
                    { label: 'ウェルネススコア', salesGrowth: 0.85, profitMargin: 0.72, employeeCount: 0.45 },
                    { label: 'エンゲージメント', salesGrowth: 0.78, profitMargin: 0.65, employeeCount: 0.32 },
                    { label: '満足度', salesGrowth: 0.82, profitMargin: 0.69, employeeCount: 0.40 },
                    { label: 'ワークライフバランス', salesGrowth: 0.71, profitMargin: 0.68, employeeCount: 0.30 }
                  ]}
                  xLabels={['売上成長率', '利益率', '従業員数']}
                />
              </Box>
            </Box>
          )}

          {analysisTabValue === 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                回帰分析
              </Typography>
              <Typography variant="body2" paragraph>
                変数間の関係性を数学的モデルで表現します
              </Typography>
              <TimeSeriesChart
                data={[
                  { date: '2022-01', wellnessScore: 70, revenue: 80 },
                  { date: '2022-02', wellnessScore: 72, revenue: 82 },
                  { date: '2022-03', wellnessScore: 75, revenue: 85 },
                  { date: '2022-04', wellnessScore: 78, revenue: 88 },
                  { date: '2022-05', wellnessScore: 80, revenue: 92 },
                  { date: '2022-06', wellnessScore: 82, revenue: 95 },
                  { date: '2022-07', wellnessScore: 85, revenue: 100 },
                  { date: '2022-08', wellnessScore: 83, revenue: 98 },
                  { date: '2022-09', wellnessScore: 84, revenue: 99 },
                  { date: '2022-10', wellnessScore: 86, revenue: 105 },
                  { date: '2022-11', wellnessScore: 87, revenue: 108 },
                  { date: '2022-12', wellnessScore: 89, revenue: 112 }
                ]}
                title="ウェルネススコアと売上の推移"
                xKey="date"
                yKeys={["wellnessScore", "revenue"]}
                labels={["ウェルネススコア", "売上（百万円）"]}
              />
            </Box>
          )}

          {analysisTabValue === 2 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                ベイズ推論
              </Typography>
              <Typography variant="body2" paragraph>
                事前確率と観測データを組み合わせた確率モデルを構築します
              </Typography>
              <BayesianChart
                data={[
                  { category: 'ウェルネススコア向上', probability: 0.85, confidence: [0.75, 0.92] },
                  { category: '売上増加', probability: 0.78, confidence: [0.68, 0.88] },
                  { category: '離職率低下', probability: 0.72, confidence: [0.62, 0.83] },
                  { category: '生産性向上', probability: 0.65, confidence: [0.55, 0.75] }
                ]}
                title="ウェルネス施策の効果予測"
              />
            </Box>
          )}

          {analysisTabValue === 3 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                クラスター分析
              </Typography>
              <Typography variant="body2" paragraph>
                データポイントを類似性に基づいてグループ化します
              </Typography>
              <ClusterChart
                data={[
                  { id: 1, x: 5, y: 8, cluster: 0, label: 'グループA' },
                  { id: 2, x: 6, y: 7, cluster: 0, label: 'グループA' },
                  { id: 3, x: 4, y: 9, cluster: 0, label: 'グループA' },
                  { id: 4, x: 7, y: 8, cluster: 0, label: 'グループA' },
                  { id: 5, x: 12, y: 3, cluster: 1, label: 'グループB' },
                  { id: 6, x: 13, y: 2, cluster: 1, label: 'グループB' },
                  { id: 7, x: 14, y: 3, cluster: 1, label: 'グループB' },
                  { id: 8, x: 12, y: 4, cluster: 1, label: 'グループB' },
                  { id: 9, x: 2, y: 2, cluster: 2, label: 'グループC' },
                  { id: 10, x: 3, y: 1, cluster: 2, label: 'グループC' },
                  { id: 11, x: 1, y: 3, cluster: 2, label: 'グループC' },
                  { id: 12, x: 2, y: 3, cluster: 2, label: 'グループC' }
                ]}
                xLabel="エンゲージメントスコア"
                yLabel="生産性スコア"
                title="従業員クラスター分析"
              />
            </Box>
          )}

          {analysisTabValue === 4 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                時系列分析
              </Typography>
              <Typography variant="body2" paragraph>
                時間経過に伴うデータの変化パターンを分析します
              </Typography>
              <TimeSeriesChart
                data={[
                  { date: '2022-01', value: 70 },
                  { date: '2022-02', value: 72 },
                  { date: '2022-03', value: 75 },
                  { date: '2022-04', value: 78 },
                  { date: '2022-05', value: 80 },
                  { date: '2022-06', value: 82 },
                  { date: '2022-07', value: 85 },
                  { date: '2022-08', value: 83 },
                  { date: '2022-09', value: 84 },
                  { date: '2022-10', value: 86 },
                  { date: '2022-11', value: 87 },
                  { date: '2022-12', value: 89 }
                ]}
                title="月別ウェルネススコア推移"
                xKey="date"
                yKeys={["value"]}
                labels={["ウェルネススコア"]}
              />
            </Box>
          )}

          {analysisTabValue === 5 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                関連分析
              </Typography>
              <Typography variant="body2" paragraph>
                項目間の関連ルールを発見します
              </Typography>
              <AssociationRulesChart
                data={[
                  { antecedent: "高エンゲージメント", consequent: "低離職率", support: 0.65, confidence: 0.82, lift: 2.3 },
                  { antecedent: "良好なワークライフバランス", consequent: "高生産性", support: 0.58, confidence: 0.75, lift: 2.1 },
                  { antecedent: "リモートワーク", consequent: "高満足度", support: 0.52, confidence: 0.70, lift: 1.9 },
                  { antecedent: "定期的なフィードバック", consequent: "成長感", support: 0.48, confidence: 0.68, lift: 1.8 },
                  { antecedent: "明確な目標設定", consequent: "目標達成", support: 0.45, confidence: 0.66, lift: 1.7 }
                ]}
                title="従業員ウェルネス関連ルール"
              />
            </Box>
          )}

          {analysisTabValue === 6 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                テキスト分析
              </Typography>
              <Typography variant="body2" paragraph>
                従業員フィードバックのテキストデータを分析します
              </Typography>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    感情分析結果
                  </Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'space-around', mb: 2 }}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="success.main">68%</Typography>
                      <Typography variant="body2">ポジティブ</Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="warning.main">22%</Typography>
                      <Typography variant="body2">中立</Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="error.main">10%</Typography>
                      <Typography variant="body2">ネガティブ</Typography>
                    </Box>
                  </Box>
                  <Typography variant="subtitle1" gutterBottom>
                    頻出キーワード
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    <Chip label="成長機会 (45)" color="primary" />
                    <Chip label="チームワーク (38)" color="primary" />
                    <Chip label="リモートワーク (32)" color="primary" />
                    <Chip label="ワークライフバランス (29)" color="primary" />
                    <Chip label="コミュニケーション (27)" color="primary" />
                    <Chip label="評価制度 (24)" color="secondary" />
                    <Chip label="福利厚生 (21)" color="primary" />
                    <Chip label="マネジメント (18)" color="secondary" />
                    <Chip label="キャリアパス (15)" color="primary" />
                    <Chip label="研修 (12)" color="primary" />
                  </Box>
                </CardContent>
              </Card>
            </Box>
          )}

          {analysisTabValue === 7 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                生存分析
              </Typography>
              <Typography variant="body2" paragraph>
                従業員の勤続年数や離職率を分析します
              </Typography>
              <SurvivalCurveChart
                data={[
                  { time: 0, survival: 1.0, group: "高ウェルネス" },
                  { time: 6, survival: 0.98, group: "高ウェルネス" },
                  { time: 12, survival: 0.95, group: "高ウェルネス" },
                  { time: 18, survival: 0.92, group: "高ウェルネス" },
                  { time: 24, survival: 0.90, group: "高ウェルネス" },
                  { time: 30, survival: 0.88, group: "高ウェルネス" },
                  { time: 36, survival: 0.85, group: "高ウェルネス" },
                  { time: 0, survival: 1.0, group: "低ウェルネス" },
                  { time: 6, survival: 0.92, group: "低ウェルネス" },
                  { time: 12, survival: 0.83, group: "低ウェルネス" },
                  { time: 18, survival: 0.72, group: "低ウェルネス" },
                  { time: 24, survival: 0.65, group: "低ウェルネス" },
                  { time: 30, survival: 0.58, group: "低ウェルネス" },
                  { time: 36, survival: 0.52, group: "低ウェルネス" }
                ]}
                title="ウェルネススコアによる従業員勤続率"
                xLabel="勤続月数"
                yLabel="残存率"
              />
            </Box>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Typography variant="h6" gutterBottom>
            レポート
          </Typography>
          <Typography variant="body1">
            企業レポートは準備中です。
          </Typography>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Typography variant="h6" gutterBottom>
            データ管理
          </Typography>
          <Typography variant="body1">
            データ管理セクションは準備中です。
          </Typography>
        </TabPanel>
      </Box>
    </Box>
  );
};

export default CompanyDetailPage;