import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  CircularProgress,
  Alert,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface StartupSurvivabilityResult {
  survival_probability: number;
  risk_factors: {
    factor: string;
    score: number;
    impact: 'high' | 'medium' | 'low';
  }[];
  survival_curve: {
    time: number;
    probability: number;
  }[];
  key_metrics: {
    metric: string;
    value: number;
    benchmark: number;
    status: 'good' | 'warning' | 'critical';
  }[];
  recommendations: {
    category: string;
    action: string;
    priority: 'high' | 'medium' | 'low';
    impact: number;
  }[];
}

const StartupSurvivabilityAnalysis: React.FC = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('5y');
  const [selectedIndustry, setSelectedIndustry] = useState<string>('all');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<StartupSurvivabilityResult | null>(null);

  const handleAnalysis = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/analysis/startup-survivability', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          timeframe: selectedTimeframe,
          industry: selectedIndustry,
        }),
      });

      if (!response.ok) {
        throw new Error('分析中にエラーが発生しました');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : '予期せぬエラーが発生しました');
    } finally {
      setLoading(false);
    }
  };

  const survivalData = result?.survival_curve.map(item => ({
    time: item.time,
    probability: item.probability * 100,
  }));

  const getStatusColor = (status: 'good' | 'warning' | 'critical') => {
    switch (status) {
      case 'good':
        return '#4caf50';
      case 'warning':
        return '#ff9800';
      case 'critical':
        return '#f44336';
      default:
        return '#000000';
    }
  };

  const getPriorityColor = (priority: 'high' | 'medium' | 'low') => {
    switch (priority) {
      case 'high':
        return '#f44336';
      case 'medium':
        return '#ff9800';
      case 'low':
        return '#4caf50';
      default:
        return '#000000';
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          スタートアップ生存可能性分析
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>分析期間</InputLabel>
              <Select
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value)}
                label="分析期間"
              >
                <MenuItem value="1y">1年</MenuItem>
                <MenuItem value="3y">3年</MenuItem>
                <MenuItem value="5y">5年</MenuItem>
                <MenuItem value="10y">10年</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>業界</InputLabel>
              <Select
                value={selectedIndustry}
                onChange={(e) => setSelectedIndustry(e.target.value)}
                label="業界"
              >
                <MenuItem value="all">全業界</MenuItem>
                <MenuItem value="tech">テクノロジー</MenuItem>
                <MenuItem value="healthcare">ヘルスケア</MenuItem>
                <MenuItem value="fintech">フィンテック</MenuItem>
                <MenuItem value="ecommerce">Eコマース</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleAnalysis}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : '分析実行'}
            </Button>
          </Grid>

          {error && (
            <Grid item xs={12}>
              <Alert severity="error">{error}</Alert>
            </Grid>
          )}

          {result && (
            <>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  生存確率: {(result.survival_probability * 100).toFixed(1)}%
                </Typography>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  生存曲線
                </Typography>
                <Box sx={{ height: 400 }}>
                  {survivalData && (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={survivalData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" label={{ value: '経過年数', position: 'bottom' }} />
                        <YAxis label={{ value: '生存確率 (%)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="probability"
                          stroke="#8884d8"
                          name="生存確率"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </Box>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  リスク要因
                </Typography>
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>要因</TableCell>
                        <TableCell align="right">スコア</TableCell>
                        <TableCell>影響度</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {result.risk_factors.map((factor, index) => (
                        <TableRow key={index}>
                          <TableCell>{factor.factor}</TableCell>
                          <TableCell align="right">
                            {(factor.score * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell>
                            <Typography
                              color={factor.impact === 'high' ? 'error' : factor.impact === 'medium' ? 'warning' : 'success'}
                            >
                              {factor.impact === 'high' ? '高' : factor.impact === 'medium' ? '中' : '低'}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  主要指標
                </Typography>
                <Grid container spacing={2}>
                  {result.key_metrics.map((metric, index) => (
                    <Grid item xs={12} md={4} key={index}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="subtitle1">
                            {metric.metric}
                          </Typography>
                          <Typography variant="h4" style={{ color: getStatusColor(metric.status) }}>
                            {metric.value.toFixed(2)}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            ベンチマーク: {metric.benchmark.toFixed(2)}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  推奨アクション
                </Typography>
                <Grid container spacing={2}>
                  {result.recommendations.map((rec, index) => (
                    <Grid item xs={12} md={6} key={index}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="subtitle1">
                            {rec.category}
                          </Typography>
                          <Typography>
                            {rec.action}
                          </Typography>
                          <Typography
                            color={getPriorityColor(rec.priority)}
                            style={{ marginTop: 8 }}
                          >
                            優先度: {rec.priority === 'high' ? '高' : rec.priority === 'medium' ? '中' : '低'}
                          </Typography>
                          <Typography color="primary">
                            期待される効果: {(rec.impact * 100).toFixed(1)}%
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Grid>
            </>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default StartupSurvivabilityAnalysis;