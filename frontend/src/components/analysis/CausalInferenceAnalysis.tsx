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
} from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface CausalInferenceResult {
  treatment_effect: number;
  confidence_interval: [number, number];
  p_value: number;
  time_series_data: {
    dates: string[];
    treated: number[];
    control: number[];
  };
}

const CausalInferenceAnalysis: React.FC = () => {
  const [selectedTreatment, setSelectedTreatment] = useState<string>('');
  const [selectedOutcome, setSelectedOutcome] = useState<string>('');
  const [selectedConfounder, setSelectedConfounder] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<CausalInferenceResult | null>(null);

  const handleAnalysis = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/analysis/causal-inference', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          treatment: selectedTreatment,
          outcome: selectedOutcome,
          confounder: selectedConfounder,
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

  const chartData = result ? {
    labels: result.time_series_data.dates,
    datasets: [
      {
        label: '介入群',
        data: result.time_series_data.treated,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
      {
        label: '対照群',
        data: result.time_series_data.control,
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1,
      },
    ],
  } : null;

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          因果推論分析
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>介入変数</InputLabel>
              <Select
                value={selectedTreatment}
                onChange={(e) => setSelectedTreatment(e.target.value)}
                label="介入変数"
              >
                <MenuItem value="wellness_program">ウェルネスプログラム</MenuItem>
                <MenuItem value="training">トレーニング</MenuItem>
                <MenuItem value="mentoring">メンタリング</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>アウトカム変数</InputLabel>
              <Select
                value={selectedOutcome}
                onChange={(e) => setSelectedOutcome(e.target.value)}
                label="アウトカム変数"
              >
                <MenuItem value="productivity">生産性</MenuItem>
                <MenuItem value="satisfaction">満足度</MenuItem>
                <MenuItem value="retention">定着率</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>交絡因子</InputLabel>
              <Select
                value={selectedConfounder}
                onChange={(e) => setSelectedConfounder(e.target.value)}
                label="交絡因子"
              >
                <MenuItem value="age">年齢</MenuItem>
                <MenuItem value="experience">経験年数</MenuItem>
                <MenuItem value="department">部署</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleAnalysis}
              disabled={loading || !selectedTreatment || !selectedOutcome || !selectedConfounder}
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
                  分析結果
                </Typography>
                <Typography>
                  介入効果: {result.treatment_effect.toFixed(2)}
                </Typography>
                <Typography>
                  信頼区間: [{result.confidence_interval[0].toFixed(2)}, {result.confidence_interval[1].toFixed(2)}]
                </Typography>
                <Typography>
                  p値: {result.p_value.toFixed(4)}
                </Typography>
              </Grid>

              <Grid item xs={12}>
                <Box sx={{ height: 400 }}>
                  {chartData && (
                    <Line
                      data={chartData}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          title: {
                            display: true,
                            text: '時系列での介入効果',
                          },
                        },
                      }}
                    />
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

export default CausalInferenceAnalysis;