import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Container,
  Card,
  CardContent,
  Avatar,
  Button,
  Chip,
  Divider,
  Stack,
  LinearProgress,
  useTheme
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import BusinessIcon from '@mui/icons-material/Business';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import AssessmentIcon from '@mui/icons-material/Assessment';
import InsightsIcon from '@mui/icons-material/Insights';
import BarChartIcon from '@mui/icons-material/BarChart';
import PieChartIcon from '@mui/icons-material/PieChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import { Link } from 'react-router-dom';
import { Company } from '@/types/company';

// recharts import
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,

  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
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
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

const HomePage: React.FC = () => {
  const theme = useTheme();

  // スコアに基づいた色を取得する関数
  const getScoreColor = (score: number) => {
    if (score >= 85) return theme.palette.success.main;
    if (score >= 75) return theme.palette.info.main;
    if (score >= 65) return theme.palette.warning.main;
    return theme.palette.error.main;
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Box sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 4,
          borderBottom: `1px solid ${theme.palette.divider}`,
          pb: 2
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Avatar sx={{ bgcolor: theme.palette.primary.main, width: 50, height: 50 }}>
              <DashboardIcon sx={{ fontSize: 28 }} />
            </Avatar>
            <Box>
              <Typography variant="h4" component="h1" fontWeight="bold">
                スタートアップウェルネスダッシュボード
              </Typography>
              <Typography variant="body2" color="text.secondary">
                企業のウェルネス状態を分析・視覚化し、改善点を提案します
              </Typography>
            </Box>
          </Box>
          <Button
            variant="contained"
            startIcon={<AssessmentIcon />}
            component={Link}
            to="/analysis"
          >
            新規分析を開始
          </Button>
        </Box>

        {/* サマリーカード */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ height: '100%', bgcolor: 'rgba(59, 130, 246, 0.08)', border: '1px solid rgba(59, 130, 246, 0.12)' }}>
              <CardContent sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: theme.palette.primary.main, mr: 1 }}>
                    <BusinessIcon />
                  </Avatar>
                  <Typography fontWeight="bold" variant="subtitle1">
                    企業数
                  </Typography>
                </Box>
                <Typography variant="h3" fontWeight="bold" sx={{ mb: 1 }}>
                  24
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  前月比 <span style={{ color: theme.palette.success.main }}>+4</span>
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ height: '100%', bgcolor: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.12)' }}>
              <CardContent sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: theme.palette.success.main, mr: 1 }}>
                    <TrendingUpIcon />
                  </Avatar>
                  <Typography fontWeight="bold" variant="subtitle1">
                    平均ウェルネススコア
                  </Typography>
                </Box>
                <Typography variant="h3" fontWeight="bold" sx={{ mb: 1 }}>
                  83
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  前月比 <span style={{ color: theme.palette.success.main }}>+2.5%</span>
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ height: '100%', bgcolor: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.12)' }}>
              <CardContent sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: theme.palette.warning.main, mr: 1 }}>
                    <AssessmentIcon />
                  </Avatar>
                  <Typography fontWeight="bold" variant="subtitle1">
                    完了分析数
                  </Typography>
                </Box>
                <Typography variant="h3" fontWeight="bold" sx={{ mb: 1 }}>
                  18
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  前月比 <span style={{ color: theme.palette.success.main }}>+5</span>
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ height: '100%', bgcolor: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.12)' }}>
              <CardContent sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ bgcolor: theme.palette.error.main, mr: 1 }}>
                    <PieChartIcon />
                  </Avatar>
                  <Typography fontWeight="bold" variant="subtitle1">
                    リスク検出数
                  </Typography>
                </Box>
                <Typography variant="h3" fontWeight="bold" sx={{ mb: 1 }}>
                  7
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  前月比 <span style={{ color: theme.palette.error.main }}>+2</span>
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Grid container spacing={3}>
          {/* ウェルネススコアのトレンド */}
          <Grid item xs={12} md={8}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TimelineIcon color="primary" />
                    <Typography variant="h6" fontWeight="bold">
                      ウェルネススコアのトレンド
                    </Typography>
                  </Box>
                  <Chip label="過去6ヶ月" size="small" />
                </Box>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={wellnessScoreData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                      <XAxis dataKey="month" />
                      <YAxis domain={[50, 100]} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: theme.palette.background.paper,
                          border: `1px solid ${theme.palette.divider}`,
                          borderRadius: 8
                        }}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="スコア"
                        stroke={theme.palette.primary.main}
                        strokeWidth={3}
                        dot={{ r: 6 }}
                        activeDot={{ r: 8 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* カテゴリ別スコア */}
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                  <PieChartIcon color="primary" />
                  <Typography variant="h6" fontWeight="bold">
                    カテゴリー別平均スコア
                  </Typography>
                </Box>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={categoryData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, value, percent }) => `${name}: ${value} (${(percent * 100).toFixed(0)}%)`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {categoryData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip
                        formatter={(value) => [`${value}点`, '値']}
                        contentStyle={{
                          backgroundColor: theme.palette.background.paper,
                          border: `1px solid ${theme.palette.divider}`,
                          borderRadius: 8
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* 相関分析 */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <InsightsIcon color="primary" />
                    <Typography variant="h6" fontWeight="bold">
                      ウェルネススコアと企業成長率の相関
                    </Typography>
                  </Box>
                  <Button
                    variant="outlined"
                    size="small"
                    component={Link}
                    to="/analysis/correlation"
                  >
                    詳細分析
                  </Button>
                </Box>
                <Box sx={{ height: 350 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart
                      margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                      <XAxis
                        type="number"
                        dataKey="wellnessScore"
                        name="ウェルネススコア"
                        domain={[60, 100]}
                        label={{ value: 'ウェルネススコア', position: 'insideBottomRight', offset: -5 }}
                      />
                      <YAxis
                        type="number"
                        dataKey="growthRate"
                        name="成長率"
                        label={{ value: '成長率 (%)', angle: -90, position: 'insideLeft' }}
                      />
                      <ZAxis type="number" dataKey="size" range={[60, 400]} />
                      <Tooltip
                        cursor={{ strokeDasharray: '3 3' }}
                        contentStyle={{
                          backgroundColor: theme.palette.background.paper,
                          border: `1px solid ${theme.palette.divider}`,
                          borderRadius: 8
                        }}
                        formatter={(value, name) => {
                          if (name === 'ウェルネススコア') return [`${value}点`, name];
                          if (name === '成長率') return [`${value}%`, name];
                          return [value, name];
                        }}
                      />
                      <Legend />
                      <Scatter
                        name="企業データ"
                        data={scatterData}
                        fill={theme.palette.primary.main}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* 最近の企業 */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                  <BusinessIcon color="primary" />
                  <Typography variant="h6" fontWeight="bold">
                    最近追加された企業
                  </Typography>
                </Box>
                <Stack spacing={2}>
                  {recentCompanies.map((company, index) => (
                    <React.Fragment key={company.id}>
                      {index > 0 && <Divider />}
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Box>
                          <Typography variant="subtitle1" fontWeight="medium">
                            {company.name}
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                            <Chip
                              label={company.industry}
                              size="small"
                              sx={{
                                borderRadius: '4px',
                                bgcolor: 'rgba(59, 130, 246, 0.1)',
                                color: 'primary.main',
                                fontWeight: 'medium'
                              }}
                            />
                            <Typography variant="body2" sx={{ mt: 1 }}>
                              ウェルネススコア:
                              <span style={{ color: getScoreColor(company.wellnessScore ?? 0), fontWeight: 'bold' }}>
                                {company.wellnessScore ?? 0}
                              </span>
                            </Typography>
                          </Box>
                        </Box>
                        <Button
                          variant="outlined"
                          size="small"
                          component={Link}
                          to={`/companies/${company.id}`}
                        >
                          詳細
                        </Button>
                      </Box>
                    </React.Fragment>
                  ))}
                </Stack>
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                  <Button
                    component={Link}
                    to="/companies"
                    endIcon={<BusinessIcon />}
                  >
                    すべての企業を表示
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* クイックアクション */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                  <BarChartIcon color="primary" />
                  <Typography variant="h6" fontWeight="bold">
                    クイックアクション
                  </Typography>
                </Box>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Button
                      variant="contained"
                      color="primary"
                      fullWidth
                      sx={{ py: 1.5, justifyContent: 'flex-start', pl: 2 }}
                      component={Link}
                      to="/companies/add"
                    >
                      <BusinessIcon sx={{ mr: 1 }} /> 新規企業を追加
                    </Button>
                  </Grid>
                  <Grid item xs={12}>
                    <Button
                      variant="contained"
                      color="secondary"
                      fullWidth
                      sx={{ py: 1.5, justifyContent: 'flex-start', pl: 2 }}
                      component={Link}
                      to="/analysis/new"
                    >
                      <AssessmentIcon sx={{ mr: 1 }} /> 新規分析を開始
                    </Button>
                  </Grid>
                  <Grid item xs={12}>
                    <Button
                      variant="outlined"
                      fullWidth
                      sx={{ py: 1.5, justifyContent: 'flex-start', pl: 2 }}
                      component={Link}
                      to="/reports"
                    >
                      <BarChartIcon sx={{ mr: 1 }} /> レポート一覧を表示
                    </Button>
                  </Grid>
                  <Grid item xs={12}>
                    <Button
                      variant="outlined"
                      fullWidth
                      sx={{ py: 1.5, justifyContent: 'flex-start', pl: 2 }}
                      component={Link}
                      to="/settings"
                    >
                      <PieChartIcon sx={{ mr: 1 }} /> 分析設定
                    </Button>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default HomePage;