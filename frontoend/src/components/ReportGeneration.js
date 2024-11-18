import React, { useState } from 'react';
import { Button, Grid, Typography, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import ReportService from '../services/ReportService';

const ReportGeneration = () => {
  const [selectedFormat, setSelectedFormat] = useState('pdf');
  const [reportGenerated, setReportGenerated] = useState(false);
  const availableFormats = ['pdf', 'excel', 'csv']; // 利用可能なレポート形式

  const handleFormatChange = (event) => {
    setSelectedFormat(event.target.value);
  };

  const handleGenerateReport = async () => {
    try {
      // レポート生成APIを呼び出す
      await ReportService.generateReport(selectedFormat);
      setReportGenerated(true); // レポート生成済みの状態を更新
      // 成功メッセージを表示する
    } catch (error) {
      // エラー処理
    }
  };

  return (
    <Grid container spacing={2} justifyContent="center">
      <Grid item xs={12}>
        <Typography variant="h6" align="center" gutterBottom>
          Report Generation
        </Typography>
      </Grid>
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel id="format-select-label">Format</InputLabel>
          <Select labelId="format-select-label" value={selectedFormat} label="Format" onChange={handleFormatChange}>
            {availableFormats.map((format) => (
              <MenuItem key={format} value={format}>
                {format.toUpperCase()}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={12} align="center">
        <Button variant="contained" color="primary" onClick={handleGenerateReport}>
          Generate Report
        </Button>
      </Grid>
      {reportGenerated && ( // レポートが生成済みであればダウンロードリンクを表示
        <Grid item xs={12} align="center">
          <Typography variant="body1" mt={2}>
            Report generated successfully. <a href={`/api/reports/${selectedFormat}`}>Download</a>
          </Typography>
        </Grid>
      )}
    </Grid>
  );
};

export default ReportGeneration;