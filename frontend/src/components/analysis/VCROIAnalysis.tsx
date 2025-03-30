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
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface VCROIResult {
  roi: number;
  irr: number;
  payback_period: number;
  investments: {
    date: string;
    amount: number;
    round: string;
  }[];
  exits: {
    date: string;
    amount: number;
    type: string;
  }[];
  portfolio_value: {
    date: string;
    value: number;
  }[];
  metrics: {
    total_invested: number;
    total_exited: number;
    active_investments: number;
    average_round_size: number;
    portfolio_health_score: number;
  };
}

const VCROIAnalysis: React.FC = () => {
  const [selectedPortfolio, setSelectedPortfolio] = useState<string>('all');
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('all');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<VCROIResult | null>(null);

  const handleAnalysis = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/analysis/vc-roi', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          portfolio: selectedPortfolio,
          timeframe: selectedTimeframe,
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

  const portfolioValueData = result?.portfolio_value.map(item => ({
    date: item.date,
    value: item.value,
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          VC投資収益率分析
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>ポートフォリオ</InputLabel>
              <Select
                value={selectedPortfolio}
                onChange={(e) => setSelectedPortfolio(e.target.value)}
                label="ポートフォリオ"
              >
                <MenuItem value="all">全ポートフォリオ</MenuItem>
                <MenuItem value="seed">シード投資</MenuItem>
                <MenuItem value="series_a">シリーズA</MenuItem>
                <MenuItem value="series_b">シリーズB</MenuItem>
                <MenuItem value="growth">成長投資</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>分析期間</InputLabel>
              <Select
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value)}
                label="分析期間"
              >
                <MenuItem value="all">全期間</MenuItem>
                <MenuItem value="1y">過去1年</MenuItem>
                <MenuItem value="3y">過去3年</MenuItem>
                <MenuItem value="5y">過去5年</MenuItem>
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
                  主要指標
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">ROI</Typography>
                        <Typography variant="h4" color="primary">
                          {(result.roi * 100).toFixed(1)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">IRR</Typography>
                        <Typography variant="h4" color="primary">
                          {(result.irr * 100).toFixed(1)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">回収期間</Typography>
                        <Typography variant="h4" color="primary">
                          {result.payback_period.toFixed(1)}年
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">ポートフォリオ健全性</Typography>
                        <Typography variant="h4" color="primary">
                          {result.metrics.portfolio_health_score.toFixed(1)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  ポートフォリオ価値推移
                </Typography>
                <Box sx={{ height: 400 }}>
                  {portfolioValueData && (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={portfolioValueData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke="#8884d8"
                          name="ポートフォリオ価値"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </Box>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  投資・エグジット概要
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>
                          投資概要
                        </Typography>
                        <Typography>
                          総投資額: ¥{result.metrics.total_invested.toLocaleString()}
                        </Typography>
                        <Typography>
                          アクティブ投資数: {result.metrics.active_investments}
                        </Typography>
                        <Typography>
                          平均ラウンドサイズ: ¥{result.metrics.average_round_size.toLocaleString()}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>
                          エグジット概要
                        </Typography>
                        <Typography>
                          総エグジット額: ¥{result.metrics.total_exited.toLocaleString()}
                        </Typography>
                        <Typography>
                          エグジット数: {result.exits.length}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Grid>
            </>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default VCROIAnalysis;