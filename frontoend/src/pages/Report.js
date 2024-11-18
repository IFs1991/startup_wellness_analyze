import React, { useState, useEffect } from 'react';
import {
  Grid,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  TextField,
} from '@mui/material';
import Navigation from '../components/Navigation';
import ReportService from '../services/ReportService';

const Report = () => {
  const [selectedStartup, setSelectedStartup] = useState('');
  const [availableStartups, setAvailableStartups] = useState([]);
  const [reportTypes, setReportTypes] = useState([]);
  const [selectedReportType, setSelectedReportType] = useState('');
  const [generatedReportUrl, setGeneratedReportUrl] = useState(null);
  const [additionalComments, setAdditionalComments] = useState(''); // レポートに追加するコメント

  useEffect(() => {
    const fetchData = async () => {
      try {
        // APIコールで利用可能なスタートアップとレポートタイプを取得する処理
        const startups = await ReportService.getAvailableStartups();
        setAvailableStartups(startups);
        setSelectedStartup(startups[0]?.id || ''); // 最初のスタートアップをデフォルトで選択

        const types = await ReportService.getAvailableReportTypes();
        setReportTypes(types);
        setSelectedReportType(types[0] || '');
      } catch (error) {
        console.error("Failed to fetch data:", error);
      }
    };
    fetchData();
  }, []);


  const handleStartupChange = (event) => {
    setSelectedStartup(event.target.value);
  };

  const handleReportTypeChange = (event) => {
    setSelectedReportType(event.target.value);
    setGeneratedReportUrl(null);
  };

  const handleAdditionalCommentsChange = (event) => {
    setAdditionalComments(event.target.value);
  };

  const handleGenerateReport = async () => {
    try {
      const reportUrl = await ReportService.generateReport({
        startupId: selectedStartup,
        reportType: selectedReportType,
        additionalComments,
      });

      setGeneratedReportUrl(reportUrl);
    } catch (error) {
      console.error('Failed to generate report:', error);
    }
  };

  return (
    <div>
      <Navigation />
      <Grid container spacing={3} mt={2}>
        <Grid item xs={12}>
          <Typography variant="h4" align="center">
            Report Generation
          </Typography>
        </Grid>
        {/* スタートアップ選択 */}
        <Grid item xs={12} md={6}>
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

        {/* レポートタイプ選択 */}
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel id="report-type-select-label">Report Type</InputLabel>
            <Select
              labelId="report-type-select-label"
              value={selectedReportType}
              label="Report Type"
              onChange={handleReportTypeChange}
            >
              {reportTypes.map((type) => (
                <MenuItem key={type} value={type}>
                  {type}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* 追加コメント入力 */}
        <Grid item xs={12}>
          <TextField
            label="Additional Comments"
            multiline
            rows={4}
            fullWidth
            value={additionalComments}
            onChange={handleAdditionalCommentsChange}
          />
        </Grid>
        {/* レポート生成ボタン */}
        <Grid item xs={12} align="center">
          <Button variant="contained" color="primary" onClick={handleGenerateReport}>
            Generate Report
          </Button>
        </Grid>

        {/* 生成されたレポートへのリンク */}
        {generatedReportUrl && (
          <Grid item xs={12} align="center">
            <Typography variant="body1" mt={2}>
              <a href={generatedReportUrl} target="_blank" rel="noopener noreferrer">
                Download Report
              </a>
            </Typography>
          </Grid>
        )}
      </Grid>
    </div>
  );
};

export default Report;