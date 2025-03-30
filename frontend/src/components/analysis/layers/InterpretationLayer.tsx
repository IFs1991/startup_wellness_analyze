import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Paper,
  Tooltip,
  IconButton,
  Divider,
} from '@mui/material';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ReferenceLine,
  Area,
  ComposedChart,
  Line,
} from 'recharts';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

interface InterpretationLayerProps {
  data: {
    trendData: {
      date: string;
      value: number;
      benchmark: number;
      confidence: {
        upper: number;
        lower: number;
      };
    }[];
    comparisonData: {
      metric: string;
      value: number;
      industryAverage: number;
      percentile: number;
      interpretation: string;
    }[];
    correlationData: {
      x: number;
      y: number;
      category: string;
      correlation: number;
      significance: number;
    }[];
    explanation: {
      summary: string;
      insights: string[];
      recommendations: string[];
    };
  };
}

const InterpretationLayer: React.FC<InterpretationLayerProps> = ({ data }) => {
  return (
    <Card>
      <CardContent>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="h6">トレンド分析</Typography>
              <Tooltip title="過去6ヶ月の推移と業界平均との比較、信頼区間付き">
                <IconButton size="small">
                  <HelpOutlineIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={data.trendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <RechartsTooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const value = payload[0]?.value;
                        const benchmark = payload[1]?.value;
                        const upperConfidence = payload[2]?.value;
                        const lowerConfidence = payload[3]?.value;

                        if (typeof value === 'number' &&
                            typeof benchmark === 'number' &&
                            typeof upperConfidence === 'number' &&
                            typeof lowerConfidence === 'number') {
                          return (
                            <Paper sx={{ p: 1 }}>
                              <Typography variant="body2">
                                日付: {payload[0].payload.date}
                              </Typography>
                              <Typography variant="body2" color="primary">
                                実績: {value.toFixed(2)}
                              </Typography>
                              <Typography variant="body2" color="secondary">
                                業界平均: {benchmark.toFixed(2)}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                信頼区間: [{lowerConfidence.toFixed(2)}, {upperConfidence.toFixed(2)}]
                              </Typography>
                            </Paper>
                          );
                        }
                      }
                      return null;
                    }}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="confidence.upper"
                    stroke="none"
                    fill="#8884d8"
                    fillOpacity={0.1}
                  />
                  <Area
                    type="monotone"
                    dataKey="confidence.lower"
                    stroke="none"
                    fill="#8884d8"
                    fillOpacity={0.1}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#8884d8"
                    name="実績"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="benchmark"
                    stroke="#82ca9d"
                    name="業界平均"
                    strokeDasharray="5 5"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Divider />
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="h6">業界比較</Typography>
              <Tooltip title="主要指標の業界平均との比較と百分位数">
                <IconButton size="small">
                  <HelpOutlineIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Grid container spacing={2}>
                {data.comparisonData.map((item, index) => (
                  <Grid item xs={12} md={6} key={index}>
                    <Box sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography variant="subtitle2">{item.metric}</Typography>
                        <Tooltip title={item.interpretation}>
                          <IconButton size="small">
                            <HelpOutlineIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="h6">{item.value.toFixed(1)}</Typography>
                        <Typography
                          variant="body2"
                          color={item.value >= item.industryAverage ? 'success.main' : 'error.main'}
                        >
                          {item.value >= item.industryAverage ? '↑' : '↓'} 業界平均: {item.industryAverage.toFixed(1)}
                        </Typography>
                      </Box>
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="body2" color="textSecondary">
                          上位{item.percentile}%以内
                        </Typography>
                      </Box>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Divider />
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="h6">相関分析</Typography>
              <Tooltip title="主要指標間の相関関係と有意性">
                <IconButton size="small">
                  <HelpOutlineIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="x" name="X値" />
                  <YAxis type="number" dataKey="y" name="Y値" />
                  <RechartsTooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <Paper sx={{ p: 1 }}>
                            <Typography variant="body2">
                              カテゴリ: {data.category}
                            </Typography>
                            <Typography variant="body2">
                              相関係数: {data.correlation.toFixed(3)}
                            </Typography>
                            <Typography variant="body2">
                              有意性: {data.significance < 0.05 ? '有意' : '非有意'}
                            </Typography>
                          </Paper>
                        );
                      }
                      return null;
                    }}
                  />
                  <Legend />
                  <Scatter
                    data={data.correlationData}
                    fill="#8884d8"
                    name="データポイント"
                  />
                  <ReferenceLine x={0} stroke="#000" />
                  <ReferenceLine y={0} stroke="#000" />
                </ScatterChart>
              </ResponsiveContainer>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Divider />
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="h6">解釈とインサイト</Typography>
              <Tooltip title="分析結果の解釈と推奨アクション">
                <IconButton size="small">
                  <HelpOutlineIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
              <Typography variant="body1" paragraph>
                {data.explanation.summary}
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                主要なインサイト:
              </Typography>
              <ul>
                {data.explanation.insights.map((insight, index) => (
                  <li key={index}>
                    <Typography variant="body2">{insight}</Typography>
                  </li>
                ))}
              </ul>
              <Typography variant="subtitle2" gutterBottom>
                推奨アクション:
              </Typography>
              <ul>
                {data.explanation.recommendations.map((recommendation, index) => (
                  <li key={index}>
                    <Typography variant="body2">{recommendation}</Typography>
                  </li>
                ))}
              </ul>
            </Paper>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default InterpretationLayer;