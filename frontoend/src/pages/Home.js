import React from 'react';
import { Typography, Grid } from '@mui/material';
import Navigation from '../components/Navigation';
import DataUpload from '../components/DataUpload'; // データアップロードコンポーネント
import DataVisualization from '../components/DataVisualization';
import ReportGeneration from '../components/ReportGeneration';
import AnalysisSettings from '../components/AnalysisSettings';


const Home = () => {
  return (
    <div>
      <Navigation />
      <Grid container spacing={3} mt={2}>
        {/* ヘッダー */}
        <Grid item xs={12}>
          <Typography variant="h4" align="center">
            Startup Wellness Data Analysis
          </Typography>
        </Grid>

        {/* データアップロード */}
        <Grid item xs={12}>
          <DataUpload />
        </Grid>

        {/* 簡単な可視化結果の表示 */}
        <Grid item xs={12}>
          <DataVisualization />
        </Grid>

        {/* レポート生成 */}
        <Grid item xs={12}>
          <ReportGeneration />
        </Grid>

        {/* 分析設定 */}
        <Grid item xs={12}>
          <AnalysisSettings />
        </Grid>
      </Grid>
    </div>
  );
};

export default Home;