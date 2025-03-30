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
import { Network } from 'react-vis-network';
import { DataSet } from 'vis-data';

interface NetworkNode {
  id: string;
  label: string;
  group: string;
  value: number;
  title: string;
}

interface NetworkEdge {
  from: string;
  to: string;
  value: number;
  title: string;
}

interface PortfolioNetworkResult {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  metrics: {
    network_density: number;
    average_degree: number;
    clustering_coefficient: number;
    centralization: number;
  };
}

const PortfolioNetworkAnalysis: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState<string>('investment_amount');
  const [timeRange, setTimeRange] = useState<[number, number]>([0, 100]);
  const [minValue, setMinValue] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<PortfolioNetworkResult | null>(null);

  const handleAnalysis = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/analysis/portfolio-network', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          metric: selectedMetric,
          time_range: timeRange,
          min_value: minValue,
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

  const networkData = result ? {
    nodes: new DataSet(result.nodes),
    edges: new DataSet(result.edges),
  } : null;

  const options = {
    nodes: {
      shape: 'dot',
      size: 30,
      font: {
        size: 12,
      },
      borderWidth: 2,
    },
    edges: {
      width: 1,
      color: { color: '#848484', highlight: '#848484', hover: '#848484' },
      smooth: {
        type: 'continuous',
      },
    },
    physics: {
      stabilization: true,
    },
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          ポートフォリオネットワーク分析
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
                <MenuItem value="investment_amount">投資額</MenuItem>
                <MenuItem value="ownership_percentage">所有比率</MenuItem>
                <MenuItem value="board_seats">取締役席数</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <Typography gutterBottom>時間範囲 (%)</Typography>
            <Slider
              value={timeRange}
              onChange={(_, newValue) => setTimeRange(newValue as [number, number])}
              valueLabelDisplay="auto"
              min={0}
              max={100}
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="最小値"
              type="number"
              value={minValue}
              onChange={(e) => setMinValue(Number(e.target.value))}
            />
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
                  ネットワーク指標
                </Typography>
                <Typography>
                  ネットワーク密度: {result.metrics.network_density.toFixed(3)}
                </Typography>
                <Typography>
                  平均次数: {result.metrics.average_degree.toFixed(2)}
                </Typography>
                <Typography>
                  クラスタリング係数: {result.metrics.clustering_coefficient.toFixed(3)}
                </Typography>
                <Typography>
                  中心性: {result.metrics.centralization.toFixed(3)}
                </Typography>
              </Grid>

              <Grid item xs={12}>
                <Box sx={{ height: 600, border: '1px solid #ddd', borderRadius: 1 }}>
                  {networkData && (
                    <Network
                      graph={networkData}
                      options={options}
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

export default PortfolioNetworkAnalysis;