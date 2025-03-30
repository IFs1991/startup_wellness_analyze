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

interface PredictiveModelResult {
  model_performance: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc_roc: number;
  };
  feature_importance: {
    feature: string;
    importance: number;
  }[];
  predictions: {
    date: string;
    actual: number;
    predicted: number;
    confidence: number;
  }[];
  model_metrics: {
    mse: number;
    rmse: number;
    mae: number;
    r2: number;
  };
}

const PredictiveModelAnalysis: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>('random_forest');
  const [selectedTarget, setSelectedTarget] = useState<string>('employee_retention');
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('1y');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<PredictiveModelResult | null>(null);

  const handleAnalysis = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/analysis/predictive-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: selectedModel,
          target: selectedTarget,
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

  const predictionData = result?.predictions.map(item => ({
    date: item.date,
    actual: item.actual,
    predicted: item.predicted,
  }));

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          予測モデル分析
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>予測モデル</InputLabel>
              <Select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                label="予測モデル"
              >
                <MenuItem value="random_forest">ランダムフォレスト</MenuItem>
                <MenuItem value="xgboost">XGBoost</MenuItem>
                <MenuItem value="lightgbm">LightGBM</MenuItem>
                <MenuItem value="neural_network">ニューラルネットワーク</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>予測対象</InputLabel>
              <Select
                value={selectedTarget}
                onChange={(e) => setSelectedTarget(e.target.value)}
                label="予測対象"
              >
                <MenuItem value="employee_retention">従業員定着率</MenuItem>
                <MenuItem value="productivity">生産性</MenuItem>
                <MenuItem value="satisfaction">満足度</MenuItem>
                <MenuItem value="performance">業績</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>予測期間</InputLabel>
              <Select
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value)}
                label="予測期間"
              >
                <MenuItem value="1m">1ヶ月</MenuItem>
                <MenuItem value="3m">3ヶ月</MenuItem>
                <MenuItem value="6m">6ヶ月</MenuItem>
                <MenuItem value="1y">1年</MenuItem>
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
                  モデル性能指標
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={2}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">精度</Typography>
                        <Typography variant="h4" color="primary">
                          {(result.model_performance.accuracy * 100).toFixed(1)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">適合率</Typography>
                        <Typography variant="h4" color="primary">
                          {(result.model_performance.precision * 100).toFixed(1)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">再現率</Typography>
                        <Typography variant="h4" color="primary">
                          {(result.model_performance.recall * 100).toFixed(1)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">F1スコア</Typography>
                        <Typography variant="h4" color="primary">
                          {(result.model_performance.f1_score * 100).toFixed(1)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={2}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">AUC-ROC</Typography>
                        <Typography variant="h4" color="primary">
                          {(result.model_performance.auc_roc * 100).toFixed(1)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  予測結果の推移
                </Typography>
                <Box sx={{ height: 400 }}>
                  {predictionData && (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={predictionData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="actual"
                          stroke="#8884d8"
                          name="実際の値"
                        />
                        <Line
                          type="monotone"
                          dataKey="predicted"
                          stroke="#82ca9d"
                          name="予測値"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </Box>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  特徴量の重要度
                </Typography>
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>特徴量</TableCell>
                        <TableCell align="right">重要度</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {result.feature_importance.map((feature, index) => (
                        <TableRow key={index}>
                          <TableCell>{feature.feature}</TableCell>
                          <TableCell align="right">
                            {(feature.importance * 100).toFixed(2)}%
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  モデル評価指標
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">MSE</Typography>
                        <Typography variant="h4" color="primary">
                          {result.model_metrics.mse.toFixed(4)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">RMSE</Typography>
                        <Typography variant="h4" color="primary">
                          {result.model_metrics.rmse.toFixed(4)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">MAE</Typography>
                        <Typography variant="h4" color="primary">
                          {result.model_metrics.mae.toFixed(4)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1">R²</Typography>
                        <Typography variant="h4" color="primary">
                          {result.model_metrics.r2.toFixed(4)}
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

export default PredictiveModelAnalysis;