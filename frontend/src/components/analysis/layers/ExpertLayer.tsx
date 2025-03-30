import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  IconButton,
  Tooltip,
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import SettingsIcon from '@mui/icons-material/Settings';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis
} from 'recharts';
import dynamic from 'next/dynamic';
import { BoxPlotChart } from '../components/BoxPlotChart';

// ForceGraph2Dの型定義
interface ForceGraphNode {
  id: string;
  label: string;
  value: number;
}

interface ForceGraphLink {
  source: string;
  target: string;
  value: number;
}

interface ForceGraphData {
  nodes: ForceGraphNode[];
  links: ForceGraphLink[];
}

interface ForceGraphProps {
  graphData: ForceGraphData;
  nodeLabel: string;
  nodeVal: string;
  linkSource: string;
  linkTarget: string;
  linkWidth: (link: ForceGraphLink) => number;
  width: number;
  height: number;
}

// SSRをオフにしたForceGraph2Dコンポーネント
const ForceGraph2D = dynamic(
  () => import('react-force-graph').then((mod) => {
    const Component = mod.ForceGraph2D;
    return Component as any;
  }),
  { ssr: false }
) as unknown as React.ComponentType<ForceGraphProps>;

const calculateQuartile = (arr: number[], q: number): number => {
  const sorted = [...arr].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  } else {
    return sorted[base];
  }
};

interface NetworkData {
  nodes: Array<{
    id: string;
    label: string;
    value: number;
  }>;
  links: Array<{
    source: string;
    target: string;
    value: number;
  }>;
}

interface ExpertLayerProps {
  data: {
    statisticalData: {
      metric: string;
      value: number;
      stdDev: number;
      pValue: number;
      confidenceInterval: [number, number];
    }[];
    modelParameters: {
      parameter: string;
      value: number;
      significance: number;
      description: string;
    }[];
    residuals: {
      predicted: number;
      actual: number;
      residual: number;
    }[];
    networkData: NetworkData;
  };
}

const ExpertLayer: React.FC<ExpertLayerProps> = ({ data }) => {
  const [selectedTab, setSelectedTab] = React.useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  };

  const handleExportData = () => {
    const exportData = {
      statisticalData: data.statisticalData,
      modelParameters: data.modelParameters,
      residuals: data.residuals,
      networkData: data.networkData,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'analysis_data.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleModelSettings = () => {
    // モデル設定処理の実装
  };

  // 残差データの準備
  const residualsData = data.residuals.map(r => r.residual);
  const min = Math.min(...residualsData);
  const max = Math.max(...residualsData);
  const q1 = calculateQuartile(residualsData, 0.25);
  const median = calculateQuartile(residualsData, 0.5);
  const q3 = calculateQuartile(residualsData, 0.75);

  // ヒストグラムデータの作成
  const createHistogramData = (data: number[], bins: number = 20) => {
    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const range = maxVal - minVal;
    const binWidth = range / bins;

    // 各ビンのカウントを初期化
    const counts = Array(bins).fill(0);

    // データポイントをビンに振り分ける
    data.forEach(val => {
      const binIndex = Math.min(Math.floor((val - minVal) / binWidth), bins - 1);
      counts[binIndex]++;
    });

    // 結果を整形
    return counts.map((count, i) => ({
      bin: (minVal + (i * binWidth) + (binWidth / 2)).toFixed(2),
      count
    }));
  };

  const histogramData = createHistogramData(residualsData);

  // 残差の散布図データ
  const scatterData = data.residuals.map((item, index) => ({
    index,
    predicted: item.predicted,
    residual: item.residual
  }));

  return (
    <Card>
      <CardContent>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography variant="h6">詳細分析</Typography>
              <Box>
                <Tooltip title="データエクスポート">
                  <IconButton onClick={handleExportData}>
                    <DownloadIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="モデル設定">
                  <IconButton onClick={handleModelSettings}>
                    <SettingsIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Tabs value={selectedTab} onChange={handleTabChange}>
              <Tab label="統計分析" />
              <Tab label="モデル詳細" />
              <Tab label="残差分析" />
              <Tab label="ネットワーク分析" />
            </Tabs>
          </Grid>

          {selectedTab === 0 && (
            <Grid item xs={12}>
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>指標</TableCell>
                      <TableCell align="right">値</TableCell>
                      <TableCell align="right">標準偏差</TableCell>
                      <TableCell align="right">p値</TableCell>
                      <TableCell align="right">信頼区間</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data.statisticalData.map((row, index) => (
                      <TableRow key={index}>
                        <TableCell>{row.metric}</TableCell>
                        <TableCell align="right">{row.value.toFixed(4)}</TableCell>
                        <TableCell align="right">{row.stdDev.toFixed(4)}</TableCell>
                        <TableCell align="right">{row.pValue.toFixed(4)}</TableCell>
                        <TableCell align="right">
                          [{row.confidenceInterval[0].toFixed(4)}, {row.confidenceInterval[1].toFixed(4)}]
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          )}

          {selectedTab === 1 && (
            <Grid item xs={12}>
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>パラメータ</TableCell>
                      <TableCell align="right">値</TableCell>
                      <TableCell align="right">有意性</TableCell>
                      <TableCell>説明</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data.modelParameters.map((row, index) => (
                      <TableRow key={index}>
                        <TableCell>{row.parameter}</TableCell>
                        <TableCell align="right">{row.value.toFixed(4)}</TableCell>
                        <TableCell align="right">{row.significance.toFixed(4)}</TableCell>
                        <TableCell>{row.description}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          )}

          {selectedTab === 2 && (
            <Grid item xs={12}>
              <Box sx={{ height: 500 }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>残差分布（ヒストグラム）</Typography>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={histogramData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="bin"
                          label={{ value: '残差値', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis
                          label={{ value: '頻度', angle: -90, position: 'insideLeft' }}
                        />
                        <RechartsTooltip />
                        <Bar dataKey="count" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>残差の散布図</Typography>
                    <ResponsiveContainer width="100%" height={200}>
                      <ScatterChart>
                        <CartesianGrid />
                        <XAxis
                          dataKey="predicted"
                          type="number"
                          name="予測値"
                          label={{ value: '予測値', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis
                          dataKey="residual"
                          type="number"
                          name="残差"
                          label={{ value: '残差', angle: -90, position: 'insideLeft' }}
                        />
                        <ZAxis range={[20, 20]} />
                        <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter name="残差値" data={scatterData} fill="#8884d8" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="subtitle1" gutterBottom>残差の箱ひげ図</Typography>
                    <BoxPlotChart
                      data={{
                        min,
                        q1,
                        median,
                        q3,
                        max
                      }}
                    />
                  </Grid>
                </Grid>
              </Box>
            </Grid>
          )}

          {selectedTab === 3 && (
            <Grid item xs={12}>
              <Box sx={{ height: 400 }}>
                <div style={{ width: '100%', height: '400px' }}>
                  {typeof window !== 'undefined' && (
                    <ForceGraph2D
                      graphData={{
                        nodes: data.networkData.nodes,
                        links: data.networkData.links
                      }}
                      nodeLabel="label"
                      nodeVal="value"
                      linkSource="source"
                      linkTarget="target"
                      linkWidth={(link: ForceGraphLink) => link.value}
                      width={800}
                      height={400}
                    />
                  )}
                </div>
              </Box>
            </Grid>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default ExpertLayer;