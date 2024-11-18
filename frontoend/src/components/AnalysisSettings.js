import React, { useState, useEffect } from 'react';
import {
  Button,
  Grid,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  TextField,
  Switch,
  FormControlLabel,
} from '@mui/material';
import AnalysisService from '../services/AnalysisService';

const AnalysisSettings = () => {
  const [googleFormQuestions, setGoogleFormQuestions] = useState([]);
  const [selectedQuestions, setSelectedQuestions] = useState([]);
  const [financialDataItems, setFinancialDataItems] = useState([]);
  const [selectedDataItems, setSelectedDataItems] = useState([]);
  const [availableAnalysisMethods, setAvailableAnalysisMethods] = useState([]);
  const [selectedAnalysisMethod, setSelectedAnalysisMethod] = useState('');
  const [availableVisualizationMethods, setAvailableVisualizationMethods] = useState([]);
  const [selectedVisualizationMethod, setSelectedVisualizationMethod] = useState('');
  const [generativeAiApiKey, setGenerativeAiApiKey] = useState('');
  const [useGenerativeAi, setUseGenerativeAi] = useState(false);
  const [generativeAiModel, setGenerativeAiModel] = useState('');

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const questions = await AnalysisService.getGoogleFormQuestions();
        setGoogleFormQuestions(questions);

        const dataItems = await AnalysisService.getFinancialDataItems();
        setFinancialDataItems(dataItems);

        const analysisMethods = await AnalysisService.getAnalysisMethods();
        setAvailableAnalysisMethods(analysisMethods);
        setSelectedAnalysisMethod(analysisMethods[0] || '');

        const visualizationMethods = await AnalysisService.getVisualizationMethods();
        setAvailableVisualizationMethods(visualizationMethods);
        setSelectedVisualizationMethod(visualizationMethods[0] || '');
      } catch (error) {
        console.error('Failed to fetch settings:', error);
      }
    };

    fetchSettings();
  }, []);

  const handleQuestionChange = (event) => {
    setSelectedQuestions(event.target.value);
  };

  const handleDataItemChange = (event) => {
    setSelectedDataItems(event.target.value);
  };

  const handleAnalysisMethodChange = (event) => {
    setSelectedAnalysisMethod(event.target.value);
  };

  const handleVisualizationMethodChange = (event) => {
    setSelectedVisualizationMethod(event.target.value);
  };

  const handleGenerativeAiApiKeyChange = (event) => {
    setGenerativeAiApiKey(event.target.value);
  };

  const handleUseGenerativeAiChange = (event) => {
    setUseGenerativeAi(event.target.checked);
  };

  const handleGenerativeAiModelChange = (event) => {
    setGenerativeAiModel(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    const settings = {
      googleFormQuestions: selectedQuestions,
      financialDataItems: selectedDataItems,
      analysisMethod: selectedAnalysisMethod,
      visualizationMethod: selectedVisualizationMethod,
      generativeAiApiKey,
      useGenerativeAi,
      generativeAiModel,
    };

    try {
      await AnalysisService.saveAnalysisSettings(settings);
      // 保存成功時の処理 (例: メッセージ表示)
      console.log('Analysis settings saved successfully.');
    } catch (error) {
      // 保存失敗時の処理 (例: エラーメッセージ表示)
      console.error('Failed to save analysis settings:', error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <Grid container spacing={2} justifyContent="center">
        <Grid item xs={12}>
          <Typography variant="h6" align="center" gutterBottom>
            Analysis Settings
          </Typography>
        </Grid>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel id="questions-select-label">Google Form Questions</InputLabel>
            <Select
              labelId="questions-select-label"
              value={selectedQuestions}
              label="Google Form Questions"
              onChange={handleQuestionChange}
              multiple
            >
              {googleFormQuestions.map((question) => (
                <MenuItem key={question.id} value={question.id}>
                  {question.text}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel id="data-items-select-label">Financial Data Items</InputLabel>
            <Select
              labelId="data-items-select-label"
              value={selectedDataItems}
              label="Financial Data Items"
              onChange={handleDataItemChange}
              multiple
            >
              {financialDataItems.map((item) => (
                <MenuItem key={item.id} value={item.id}>
                  {item.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel id="analysis-method-select-label">Analysis Method</InputLabel>
            <Select
              labelId="analysis-method-select-label"
              value={selectedAnalysisMethod}
              label="Analysis Method"
              onChange={handleAnalysisMethodChange}
            >
              {availableAnalysisMethods.map((method) => (
                <MenuItem key={method} value={method}>
                  {method}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel id="visualization-method-select-label">Visualization Method</InputLabel>
            <Select
              labelId="visualization-method-select-label"
              value={selectedVisualizationMethod}
              label="Visualization Method"
              onChange={handleVisualizationMethodChange}
            >
              {availableVisualizationMethods.map((method) => (
                <MenuItem key={method} value={method}>
                  {method}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={8}>
          <TextField
            label="Generative AI API Key"
            fullWidth
            value={generativeAiApiKey}
            onChange={handleGenerativeAiApiKeyChange}
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <FormControlLabel
            control={<Switch checked={useGenerativeAi} onChange={handleUseGenerativeAiChange} />}
            label="Use Generative AI"
          />
        </Grid>
        {useGenerativeAi && (
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel id="generative-ai-model-select-label">Generative AI Model</InputLabel>
              <Select
                labelId="generative-ai-model-select-label"
                value={generativeAiModel}
                label="Generative AI Model"
                onChange={handleGenerativeAiModelChange}
              >
                {['gpt-3.5-turbo', 'text-davinci-003'].map((model) => ( // 利用可能な生成AIモデル
                  <MenuItem key={model} value={model}>
                    {model}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        )}
        <Grid item xs={12} align="center">
          <Button type="submit" variant="contained" color="primary">
            Save Settings
          </Button>
        </Grid>
      </Grid>
    </form>
  );
};

export default AnalysisSettings;