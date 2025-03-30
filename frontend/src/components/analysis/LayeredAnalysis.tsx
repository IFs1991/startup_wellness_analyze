import React, { useState } from 'react';
import {
  Box,
  Tabs,
  Tab,
  Typography,
  Paper,
  Grid,
  Button,
  Tooltip,
  IconButton,
} from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import SummaryLayer from './layers/SummaryLayer';
import InterpretationLayer from './layers/InterpretationLayer';
import ExpertLayer from './layers/ExpertLayer';

interface LayeredAnalysisProps {
  data: {
    summary: {
      score: number;
      label: string;
      previousValue: number;
      industryAverage: number;
      grade: string;
      status: 'excellent' | 'good' | 'needs_improvement';
    };
    interpretation: {
      trendData: {
        date: string;
        value: number;
        benchmark: number;
        confidence: number;
      }[];
      comparisonData: {
        metric: string;
        value: number;
        industryAverage: number;
        percentile: number;
      }[];
      correlationData: {
        x: number;
        y: number;
        category: string;
      }[];
      explanation: string;
    };
    expert: {
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
      networkData: {
        nodes: {
          id: string;
          label: string;
          value: number;
        }[];
        edges: {
          source: string;
          target: string;
          value: number;
        }[];
      };
    };
  };
}

const LayeredAnalysis: React.FC<LayeredAnalysisProps> = ({ data }) => {
  const [selectedLayer, setSelectedLayer] = useState(0);

  const handleLayerChange = (event: React.SyntheticEvent, newValue: number) => {
    setSelectedLayer(newValue);
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h5">分析結果</Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Tooltip title="レイヤー別の分析結果を表示します。ユーザーの知識レベルに応じて適切な詳細度で情報を提供します。">
            <IconButton size="small">
              <HelpOutlineIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={selectedLayer}
          onChange={handleLayerChange}
          variant="fullWidth"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography>概要</Typography>
                <Tooltip title="主要な指標と評価を簡潔に表示します">
                  <IconButton size="small">
                    <HelpOutlineIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
          <Tab
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography>詳細</Typography>
                <Tooltip title="トレンド分析、業界比較、相関分析などの詳細情報を表示します">
                  <IconButton size="small">
                    <HelpOutlineIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
          <Tab
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography>専門分析</Typography>
                <Tooltip title="統計分析、モデル詳細、残差分析などの専門的な情報を表示します">
                  <IconButton size="small">
                    <HelpOutlineIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
        </Tabs>
      </Paper>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          {selectedLayer === 0 && <SummaryLayer data={data.summary} />}
          {selectedLayer === 1 && <InterpretationLayer data={data.interpretation} />}
          {selectedLayer === 2 && <ExpertLayer data={data.expert} />}
        </Grid>
      </Grid>
    </Box>
  );
};

export default LayeredAnalysis;