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
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface HealthInvestmentEffectResult {
  index_value: number;
  components: {
    name: string;
    value: number;
    weight: number;
  }[];
  time_series: {
    date: string;
    index: number;
    components: {
      [key: string]: number;
    };
  }[];
  recommendations: {
    category: string;
    action: string;
    potential_impact: number;
  }[];
}

const HealthInvestmentEffectAnalysis: React.FC = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('1y');
  const [selectedDepartment, setSelectedDepartment] = useState<string>('all');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<HealthInvestmentEffectResult | null>(null);

  const handleAnalysis = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/analysis/health-investment-effect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          timeframe: selectedTimeframe,
          department: selectedDepartment,
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

  const chartData = result?.time_series.map(item => ({
    date: item.date,
    index: item.index,
    ...item.components,
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          健康投資効果指数分析
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
                <MenuItem value="1m">1ヶ月</MenuItem>
                <MenuItem value="3m">3ヶ月</MenuItem>
                <MenuItem value="6m">6ヶ月</MenuItem>
                <MenuItem value="1y">1年</MenuItem>
                <MenuItem value="3y">3年</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>部署</InputLabel>
              <Select
                value={selectedDepartment}
                onChange={(e) => setSelectedDepartment(e.target.value)}
                label="部署"
              >
                <MenuItem value="all">全社</MenuItem>
                <MenuItem value="engineering">エンジニアリング</MenuItem>
                <MenuItem value="sales">営業</MenuItem>
                <MenuItem value="marketing">マーケティング</MenuItem>
                <MenuItem value="hr">人事</MenuItem>
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
                  健康投資効果指数: {result.index_value.toFixed(2)}
                </Typography>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  構成要素
                </Typography>
                <Grid container spacing={2}>
                  {result.components.map((component, index) => (
                    <Grid item xs={12} md={6} key={index}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="subtitle1">
                            {component.name}
                          </Typography>
                          <Typography>
                            値: {component.value.toFixed(2)}
                          </Typography>
                          <Typography>
                            重み: {component.weight.toFixed(2)}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  時系列推移
                </Typography>
                <Box sx={{ height: 400 }}>
                  {chartData && (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="index" fill="#8884d8" name="健康投資効果指数" />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </Box>
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
                          <Typography color="primary">
                            期待される効果: {rec.potential_impact.toFixed(2)}
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

export default HealthInvestmentEffectAnalysis;