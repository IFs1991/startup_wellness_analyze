import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Chip,
  Tooltip,
} from '@mui/material';
import {
  RadialBarChart,
  RadialBar,
  ResponsiveContainer,
  PolarGrid,
} from 'recharts';

interface SummaryLayerProps {
  data: {
    score: number;
    label: string;
    previousValue: number;
    industryAverage: number;
    grade: string;
    status: 'excellent' | 'good' | 'needs_improvement';
    impactBadge: {
      grade: string;
      label: string;
    };
  };
}

const SummaryLayer: React.FC<SummaryLayerProps> = ({ data }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent':
        return '#4caf50';
      case 'good':
        return '#ff9800';
      case 'needs_improvement':
        return '#f44336';
      default:
        return '#757575';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'excellent':
        return '優秀';
      case 'good':
        return '良好';
      case 'needs_improvement':
        return '要改善';
      default:
        return '';
    }
  };

  const gaugeData = [
    {
      name: 'スコア',
      value: data.score,
      fill: getStatusColor(data.status),
    },
  ];

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">{data.label}</Typography>
              <Tooltip title={`${getStatusText(data.status)}：${data.score}点`}>
                <Chip
                  label={getStatusText(data.status)}
                  color={data.status === 'excellent' ? 'success' : data.status === 'good' ? 'warning' : 'error'}
                  size="small"
                />
              </Tooltip>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ position: 'relative', height: 120 }}>
              <ResponsiveContainer width="100%" height="100%">
                <RadialBarChart
                  cx="50%"
                  cy="50%"
                  innerRadius="60%"
                  outerRadius="80%"
                  barSize={10}
                  data={gaugeData}
                  startAngle={180}
                  endAngle={0}
                >
                  <PolarGrid />
                  <RadialBar
                    background
                    dataKey="value"
                    cornerRadius={30}
                    fill={getStatusColor(data.status)}
                  />
                  <text
                    x="50%"
                    y="50%"
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="progress-label"
                    fill="#666"
                    fontSize={20}
                  >
                    {data.score}
                  </text>
                </RadialBarChart>
              </ResponsiveContainer>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Typography variant="body2" color="textSecondary">
                前回比:
              </Typography>
              <Typography
                variant="body2"
                color={data.score >= data.previousValue ? 'success.main' : 'error.main'}
              >
                {data.score >= data.previousValue ? '+' : '-'}
                {Math.abs(data.score - data.previousValue).toFixed(1)}%
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              業界平均: {data.industryAverage.toFixed(1)}
            </Typography>
            <Tooltip title={`業界平均との比較: ${((data.score / data.industryAverage) * 100).toFixed(1)}%`}>
              <LinearProgress
                variant="determinate"
                value={(data.score / data.industryAverage) * 100}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: '#e0e0e0',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: getStatusColor(data.status),
                    borderRadius: 4,
                  },
                }}
              />
            </Tooltip>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
              <Tooltip title={data.impactBadge.label}>
                <Typography variant="h4" color="primary">
                  {data.impactBadge.grade}
                </Typography>
              </Tooltip>
              <Typography variant="body2" color="textSecondary">
                評価
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default SummaryLayer;