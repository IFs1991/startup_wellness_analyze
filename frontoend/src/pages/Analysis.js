import React, { useState, useEffect } from 'react';
import { Grid, Typography, Button, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import NoteInput from '../components/NoteInput';
import Navigation from '../components/Navigation';
import AnalysisService from '../services/AnalysisService';

const Analysis = () => {
  const [analysisResults, setAnalysisResults] = useState([]);
  const [selectedStartup, setSelectedStartup] = useState('');
  const [availableStartups, setAvailableStartups] = useState([]);
  const [selectedAnalysisType, setSelectedAnalysisType] = useState('Descriptive Statistics');
  const [isLoading, setIsLoading] = useState(false);
  const analysisTypes = ['Descriptive Statistics', 'Correlation Analysis', 'Time Series Analysis', 'Cluster Analysis', 'PCA', 'Survival Analysis', 'Association Analysis', 'Text Analysis'];

  useEffect(() => {
    const fetchStartups = async () => {
      try {
        const startups = await AnalysisService.getAvailableStartups();
        setAvailableStartups(startups);
        setSelectedStartup(startups[0] || '');
      } catch (error) {
        console.error('Failed to fetch startups:', error);
      }
    };

    fetchStartups();
  }, []);

  const handleStartupChange = (event) => {
    setSelectedStartup(event.target.value);
  };

  const handleAnalysisTypeChange = (event) => {
    setSelectedAnalysisType(event.target.value);
  };

  const handleRunAnalysis = async () => {
    setIsLoading(true);
    try {
      const data = await AnalysisService.runAnalysis(selectedStartup, selectedAnalysisType);
      setAnalysisResults(data);
    } catch (error) {
      console.error('Failed to run analysis:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // 分析結果の表示部分は、selectedAnalysisType に基づいて適切なコンポーネントを表示する
  const renderAnalysisResults = () => {
    // switch文でselectedAnalysisTypeに応じた表示を切り替える例
    switch (selectedAnalysisType) {
      case 'Descriptive Statistics':
        return (
          // 記述統計量を表示
          <div>
            {/* 分析結果を表示するコンポーネント */}
          </div>
        );
      case 'Correlation Analysis':
        return (
          // 相関分析の結果を表示
          <div>
            {/* 分析結果を表示するコンポーネント */}
          </div>
        );
      // 他の分析タイプについても同様に追加
      default:
        return null;
    }
  };


  return (
    <div>
      <Navigation />
      <Grid container spacing={3} mt={2}>
        <Grid item xs={12}>
          <Typography variant="h4" align="center">
            Analysis
          </Typography>
        </Grid>
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel id="startup-select-label">Select Startup</InputLabel>
            <Select labelId="startup-select-label" value={selectedStartup} label="Select Startup" onChange={handleStartupChange}>
              {availableStartups.map((startup) => (
                <MenuItem key={startup.id} value={startup.id}>
                  {startup.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel id="analysis-type-select-label">Analysis Type</InputLabel>
            <Select labelId="analysis-type-select-label" value={selectedAnalysisType} label="Analysis Type" onChange={handleAnalysisTypeChange}>
              {analysisTypes.map((type) => (
                <MenuItem key={type} value={type}>
                  {type}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={4} align="center">
          <Button variant="contained" color="primary" onClick={handleRunAnalysis} disabled={isLoading}>
            {isLoading ? 'Running...' : 'Run Analysis'}
          </Button>
        </Grid>
        {/* 分析結果表示 */}
        <Grid item xs={12}>
          {renderAnalysisResults()}
        </Grid>
        <Grid item xs={12}>
          <NoteInput analysisId={/* 分析IDを渡す */} />
        </Grid>
      </Grid>
    </div>
  );
};

export default Analysis;