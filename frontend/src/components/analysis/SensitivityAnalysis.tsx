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
  Slider,
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

interface SensitivityResult {
  base_value: number;
  sensitivity_scores: {
    parameter: string;
    score: number;
    impact: 'positive' | 'negative';
  }[];
  parameter_ranges: {
    parameter: string;
    min: number;
    max: number;
    current: number;
  }[];
  scenario_analysis: {
    scenario: string;
    value: number;
    probability: number;
  }[];
}

const SensitivityAnalysis: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState<string>('employee_retention');
  const [selectedParameters, setSelectedParameters] = useState<string[]>([]);
  const [parameterRanges, setParameterRanges] = useState<{
    [key: string]: { min: number; max: number; current: number };
  }>({});
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<SensitivityResult | null>(null);

  const handleAnalysis = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/analysis/sensitivity', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          metric: selectedMetric,
          parameters: selectedParameters,
          parameter_ranges: parameterRanges,
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

  const handleParameterRangeChange = (
    parameter: string,
    field: 'min' | 'max' | 'current',
    value: number
  ) => {
    setParameterRanges(prev => ({
      ...prev,
      [parameter]: {
        ...prev[parameter],
        [field]: value,
      },
    }));
  };

  const sensitivityData = result?.sensitivity_scores.map(item => ({
    parameter: item.parameter,
    score: Math.abs(item.score),
    impact: item.impact,
  }));

  const scenarioData = result?.scenario_analysis.map(item => ({
    scenario: item.scenario,
    value: item.value,
    probability: item.probability * 100,
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          感度分析
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
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

          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>分析パラメータ</InputLabel>
              <Select
                multiple
                value={selectedParameters}
                onChange={(e) => setSelectedParameters(e.target.value as string[])}
                label="分析パラメータ"
              >
                <MenuItem value="wellness_program">ウェルネスプログラム</MenuItem>
                <MenuItem value="training">トレーニング</MenuItem>
                <MenuItem value="mentoring">メンタリング</MenuItem>
                <MenuItem value="work_environment">職場環境</MenuItem>
                <MenuItem value="compensation">報酬</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          {selectedParameters.map((parameter) => (
            <Grid item xs={12} key={parameter}>
              <Typography gutterBottom>{parameter}</Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="最小値"
                    type="number"
                    value={parameterRanges[parameter]?.min || 0}
                    onChange={(e) =>
                      handleParameterRangeChange(parameter, 'min', Number(e.target.value))
                    }
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="現在値"
                    type="number"
                    value={parameterRanges[parameter]?.current || 0}
                    onChange={(e) =>
                      handleParameterRangeChange(parameter, 'current', Number(e.target.value))
                    }
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="最大値"
                    type="number"
                    value={parameterRanges[parameter]?.max || 0}
                    onChange={(e) =>
                      handleParameterRangeChange(parameter, 'max', Number(e.target.value))
                    }
                  />
                </Grid>
              </Grid>
            </Grid>
          ))}

          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleAnalysis}
              disabled={loading || selectedParameters.length === 0}
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
                  基本値: {result.base_value.toFixed(2)}
                </Typography>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  感度スコア
                </Typography>
                <Box sx={{ height: 400 }}>
                  {sensitivityData && (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={sensitivityData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="parameter" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar
                          dataKey="score"
                          fill="#8884d8"
                          name="感度スコア"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </Box>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  シナリオ分析
                </Typography>
                <Box sx={{ height: 400 }}>
                  {scenarioData && (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={scenarioData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="scenario" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar
                          dataKey="value"
                          fill="#82ca9d"
                          name="予測値"
                        />
                        <Bar
                          dataKey="probability"
                          fill="#8884d8"
                          name="確率 (%)"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </Box>
              </Grid>
            </>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default SensitivityAnalysis;