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
  Slider,
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

interface MonteCarloResult {
  simulations: {
    iteration: number;
    value: number;
  }[];
  statistics: {
    mean: number;
    median: number;
    std_dev: number;
    min: number;
    max: number;
    percentile_5: number;
    percentile_95: number;
  };
  confidence_intervals: {
    level: number;
    lower: number;
    upper: number;
  }[];
}

const MonteCarloSimulation: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState<string>('employee_retention');
  const [numSimulations, setNumSimulations] = useState<number>(1000);
  const [timeHorizon, setTimeHorizon] = useState<number>(12);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<MonteCarloResult | null>(null);

  const handleAnalysis = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/analysis/monte-carlo', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          metric: selectedMetric,
          num_simulations: numSimulations,
          time_horizon: timeHorizon,
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

  const simulationData = result?.simulations.map(item => ({
    iteration: item.iteration,
    value: item.value,
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          モンテカルロシミュレーション
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>分析指標</InputLabel>
              <Select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value)}
                label="分析指標"
              >
                <MenuItem value="employee_retention">従業員定着率</MenuItem>
                <MenuItem value="productivity">生産性</MenuItem>
                <MenuItem value="satisfaction">満足度</MenuItem>
                <MenuItem value="performance">業績</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <Typography gutterBottom>シミュレーション回数</Typography>
            <Slider
              value={numSimulations}
              onChange={(_, newValue) => setNumSimulations(newValue as number)}
              min={100}
              max={10000}
              step={100}
              valueLabelDisplay="auto"
            />
            <Typography variant="body2" color="textSecondary">
              {numSimulations}回
            </Typography>
          </Grid>

          <Grid item xs={12} md={4}>
            <Typography gutterBottom>予測期間（ヶ月）</Typography>
            <Slider
              value={timeHorizon}
              onChange={(_, newValue) => setTimeHorizon(newValue as number)}
              min={1}
              max={60}
              step={1}
              valueLabelDisplay="auto"
            />
            <Typography variant="body2" color="textSecondary">
              {timeHorizon}ヶ月
            </Typography>
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
                  統計指標
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">平均値</Typography>
                        <Typography variant="h4" color="primary">
                          {result.statistics.mean.toFixed(2)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">中央値</Typography>
                        <Typography variant="h4" color="primary">
                          {result.statistics.median.toFixed(2)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">標準偏差</Typography>
                        <Typography variant="h4" color="primary">
                          {result.statistics.std_dev.toFixed(2)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">範囲</Typography>
                        <Typography variant="h4" color="primary">
                          {result.statistics.min.toFixed(2)} - {result.statistics.max.toFixed(2)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  シミュレーション結果
                </Typography>
                <Box sx={{ height: 400 }}>
                  {simulationData && (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={simulationData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="iteration" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke="#8884d8"
                          name="シミュレーション値"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </Box>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  信頼区間
                </Typography>
                <Grid container spacing={2}>
                  {result.confidence_intervals.map((interval, index) => (
                    <Grid item xs={12} md={4} key={index}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="subtitle1">
                            {interval.level * 100}%信頼区間
                          </Typography>
                          <Typography>
                            {interval.lower.toFixed(2)} - {interval.upper.toFixed(2)}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  分位点
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">5パーセンタイル</Typography>
                        <Typography variant="h4" color="primary">
                          {result.statistics.percentile_5.toFixed(2)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">95パーセンタイル</Typography>
                        <Typography variant="h4" color="primary">
                          {result.statistics.percentile_95.toFixed(2)}
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

export default MonteCarloSimulation;