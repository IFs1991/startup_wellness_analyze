import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useCompanyData, CorrelationData as HookCorrelationData, SurvivalAnalysisData } from '@/hooks/useCompanyData';
import { ArrowLeft, RefreshCw } from 'lucide-react';
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
  Chip,
  CircularProgress,
  Alert,
  Paper
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import EditIcon from '@mui/icons-material/Edit';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {
  ClusterChart,
  TimeSeriesChart,
  AssociationRulesChart,
  SurvivalCurveChart,
  CorrelationMatrix
} from '../components/charts';
import { SurvivalCurve } from '../components/charts/SurvivalCurveChart';
import { CompanyInfo, EmployeeMetrics, ClusterData, TimeSeriesData as HookTimeSeriesData, AssociationRule, TextAnalysisResult } from '@/hooks/useCompanyData';
import { TimeSeriesData as ChartTimeSeriesData } from '../components/charts/TimeSeriesChart';

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
        <Box sx={{ p: 3, minHeight: 400 }}>
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
  const {
    companyInfo,
    employeeMetrics,
    clusterData,
    timeSeriesData,
    associationRules,
    textAnalysis,
    correlationData,
    survivalData,
    loading,
    error,
    refreshData
  } = useCompanyData(id || '');

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleAnalysisTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setAnalysisTabValue(newValue);
  };

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
  const formattedCorrelationPoints = formatCorrelationDataForChart(correlationData);

  const formatSurvivalDataForChart = (data: SurvivalAnalysisData | undefined): SurvivalCurve[] | undefined => {
    if (!data || !data.segments) return undefined;
    return data.segments.map((seg) => ({
      name: seg.name,
      color: seg.color,
      data: seg.data
    }));
  };
  const formattedSurvivalCurves = formatSurvivalDataForChart(survivalData);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh', bgcolor: 'grey.900', color: 'white' }}>
        <CircularProgress color="inherit" />
        <Typography sx={{ ml: 2 }}>企業データを読み込んでいます...</Typography>
      </Box>
    );
  }

  if (error || !companyInfo) {
    return (
      <Box sx={{ p: 3, bgcolor: 'grey.900', minHeight: '100vh', color: 'white' }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/companies')}
          variant="outlined"
          color="inherit"
          sx={{ mb: 3 }}
        >
          企業一覧に戻る
        </Button>
        <Alert severity="error">
          {error ? `データの読み込みに失敗しました: ${error.message}` : '企業情報が見つかりませんでした。'}
          <button onClick={refreshData} style={{ marginLeft: '1em', padding: '0.2em 0.5em' }}>再試行</button>
        </Alert>
      </Box>
    );
  }

  const renderOverview = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>基本情報</Typography>
            <Typography><strong>産業:</strong> {companyInfo.industry}</Typography>
            <Typography><strong>設立日:</strong> {companyInfo.foundingDate ? new Date(companyInfo.foundingDate).toLocaleDateString() : 'N/A'}</Typography>
            <Typography><strong>所在地:</strong> {companyInfo.location}</Typography>
            <Typography><strong>従業員数:</strong> {companyInfo.employeeCount}</Typography>
            <Typography><strong>資金調達ステージ:</strong> {companyInfo.fundingStage}</Typography>
            <Typography variant="body2" sx={{ mt: 2 }}>{companyInfo.description}</Typography>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>主要メトリクス</Typography>
            {employeeMetrics ? (
              <>
                <Typography><strong>ウェルネススコア:</strong> {employeeMetrics.wellnessScore?.toFixed(1)} (前期間比: {employeeMetrics.wellnessScoreChange >= 0 ? '+' : ''}{employeeMetrics.wellnessScoreChange?.toFixed(1)}%)</Typography>
                <Typography><strong>エンゲージメント率:</strong> {employeeMetrics.engagementRate?.toFixed(1)}%</Typography>
                <Typography><strong>離職率:</strong> {employeeMetrics.turnoverRate?.toFixed(1)}%</Typography>
                <Typography><strong>生産性スコア:</strong> {employeeMetrics.productivityScore?.toFixed(1)}</Typography>
                <Typography><strong>満足度スコア:</strong> {employeeMetrics.satisfactionScore?.toFixed(1)}</Typography>
                <Typography><strong>メンタルヘルススコア:</strong> {employeeMetrics.mentalHealthScore?.toFixed(1)}</Typography>
                <Typography><strong>ワークライフバランススコア:</strong> {employeeMetrics.workLifeBalanceScore?.toFixed(1)}</Typography>
              </>
            ) : (
              <Typography>従業員メトリクスはありません。</Typography>
            )}
          </CardContent>
        </Card>
      </Grid>
      {textAnalysis && (
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>テキスト分析サマリー</Typography>
              <Typography><strong>感情分析 (Positive):</strong> {(textAnalysis.sentiment?.positive * 100).toFixed(1)}%</Typography>
              <Typography><strong>トップキーワード:</strong> {textAnalysis.topKeywords?.map(k => k.word).join(', ')}</Typography>
              <Typography sx={{ mt: 1 }}><strong>インサイト:</strong> {textAnalysis.insightSummary}</Typography>
            </CardContent>
          </Card>
        </Grid>
      )}
    </Grid>
  );

  const formatTimeSeriesDataForChart = (data: HookTimeSeriesData | undefined): ChartTimeSeriesData | undefined => {
    if (!data || !data.labels || !data.datasets) return undefined;
    return {
      dates: data.labels,
      series: data.datasets.map(ds => ({
        name: ds.label,
        data: ds.data,
        color: ds.borderColor,
        isMainMetric: ds.label.includes('スコア'),
        scale: 1
      })),
      annotations: [],
      insights: []
    };
  };
  const formattedTimeSeriesData = formatTimeSeriesDataForChart(timeSeriesData);

  const renderAnalysis = () => (
    <Box>
      <Paper sx={{ mb: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Tabs
          value={analysisTabValue}
          onChange={handleAnalysisTabChange}
          aria-label="分析サブタブ"
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="クラスター分析" {...a11yProps(0)} />
          <Tab label="時系列分析" {...a11yProps(1)} />
          <Tab label="相関分析" {...a11yProps(2)} />
          <Tab label="生存時間分析" {...a11yProps(3)} />
          <Tab label="アソシエーション分析" {...a11yProps(4)} />
        </Tabs>
      </Paper>

      <TabPanel value={analysisTabValue} index={0}>
        {clusterData ? (
          <ClusterChart
            data={clusterData.points}
            clusters={clusterData.clusters}
            selectedCluster={null}
            onClusterSelect={(clusterId) => console.log('Selected Cluster:', clusterId)}
            xAxisLabel={clusterData.xAxisLabel}
            yAxisLabel={clusterData.yAxisLabel}
          />
        ) : <Typography>クラスター分析データはありません。</Typography>}
      </TabPanel>
      <TabPanel value={analysisTabValue} index={1}>
        {formattedTimeSeriesData ? <TimeSeriesChart data={formattedTimeSeriesData} /> : <Typography>時系列分析データはありません。</Typography>}
      </TabPanel>
      <TabPanel value={analysisTabValue} index={2}>
        {formattedCorrelationPoints ? (
          <Box sx={{ height: 500 }}>
            <CorrelationMatrix data={formattedCorrelationPoints} />
          </Box>
        ) : <Typography>相関分析データはありません。</Typography>}
      </TabPanel>
      <TabPanel value={analysisTabValue} index={3}>
        {formattedSurvivalCurves ? (
          <Box sx={{ height: 400 }}>
            <SurvivalCurveChart curves={formattedSurvivalCurves} />
          </Box>
        ) : <Typography>生存分析データはありません。</Typography>}
      </TabPanel>
      <TabPanel value={analysisTabValue} index={4}>
        {associationRules ? <AssociationRulesChart rules={associationRules} /> : <Typography>アソシエーション分析データはありません。</Typography>}
      </TabPanel>
    </Box>
  );

  return (
    <Box sx={{ p: { xs: 1, sm: 2, md: 3 } }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3, flexWrap: 'wrap' }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/companies')}
          variant="outlined"
          sx={{ mr: 1, mb: { xs: 1, sm: 0 } }}
        >
          戻る
        </Button>
        <Typography variant="h4" component="h1" sx={{ flexGrow: 1, textAlign: { xs: 'center', sm: 'left' }, order: { xs: 3, sm: 2 }, my: { xs: 1, sm: 0 } }}>
          {companyInfo.name}
        </Typography>
        <Box sx={{ order: { xs: 2, sm: 3 } }}>
          <Button
            startIcon={<RefreshCw size={18} />}
            onClick={refreshData}
            variant="outlined"
            sx={{ mr: 1 }}
            disabled={loading}
          >
            更新
          </Button>
          <Button
            startIcon={<EditIcon />}
            onClick={() => navigate(`/companies/${id}/edit`)}
            variant="contained"
            disabled={loading}
          >
            編集
          </Button>
        </Box>
      </Box>

      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1" sx={{ flexGrow: 1 }}>
          {companyInfo.name}
        </Typography>
        {companyInfo.industry && <Chip label={companyInfo.industry} color="primary" sx={{ mr: 1 }} />}
        {companyInfo.fundingStage && <Chip label={companyInfo.fundingStage} color="secondary" />}
      </Box>

      {employeeMetrics && (
        <Typography variant="body1" sx={{ mb: 3 }}>
          {companyInfo.name}のウェルネススコアは{employeeMetrics.wellnessScore?.toFixed(1)}点です。
        </Typography>
      )}

      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="企業詳細タブ"
        >
          <Tab label="概要" {...a11yProps(0)} />
          <Tab label="分析" {...a11yProps(1)} />
        </Tabs>
      </Paper>

      <TabPanel value={tabValue} index={0}>
        {renderOverview()}
      </TabPanel>
      <TabPanel value={tabValue} index={1}>
        {renderAnalysis()}
      </TabPanel>
    </Box>
  );
};

export default CompanyDetailPage;