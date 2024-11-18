import React, { useState, useEffect } from 'react';
import {
  Grid,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Checkbox,
  ListItemText,
  OutlinedInput,
  Slider,
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import Navigation from '../components/Navigation';
import VisualizationService from '../services/VisualizationService';

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;
const MenuProps = {
  PaperProps: {
    style: {
      maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
      width: 250,
    },
  },
};

const Visualization = () => {
  const [selectedStartup, setSelectedStartup] = useState('');
  const [availableStartups, setAvailableStartups] = useState([]);
  const [selectedVisualizationType, setSelectedVisualizationType] = useState('');
  const [availableVisualizationTypes, setAvailableVisualizationTypes] = useState([]);
  const [visualizationData, setVisualizationData] = useState([]);
  const [availableDataPoints, setAvailableDataPoints] = useState([]);
  const [selectedDataPoints, setSelectedDataPoints] = useState([]);
  const [timeRange, setTimeRange] = useState([2015, 2025]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // 利用可能なスタートアップを取得
        const startups = await VisualizationService.getAvailableStartups();
        setAvailableStartups(startups);
        setSelectedStartup(startups[0]?.id || ''); // 最初のスタートアップを選択状態に

        // 利用可能な可視化タイプを取得
        const visualizationTypes = await VisualizationService.getAvailableVisualizationTypes();
        setAvailableVisualizationTypes(visualizationTypes);
        setSelectedVisualizationType(visualizationTypes[0] || '');
      } catch (error) {
        console.error('Failed to fetch data:', error);
      }
    };

    fetchData();
  }, []);


  useEffect(() => {
    const fetchVisualizationData = async () => {
      if (selectedStartup && selectedVisualizationType) {
        try {
          // 可視化データを取得
          const data = await VisualizationService.getVisualizationData({
            startupId: selectedStartup,
            visualizationType: selectedVisualizationType,
            dataPoints: selectedDataPoints,
            timeRange: timeRange,
          });
          setVisualizationData(data);
          setAvailableDataPoints(Object.keys(data[0] || {}));
          setSelectedDataPoints(Object.keys(data[0] || {}).slice(1));
        } catch (error) {
          console.error("Failed to fetch visualization data:", error);
        }
      }
    };

    fetchVisualizationData();
  }, [selectedStartup, selectedVisualizationType, selectedDataPoints, timeRange]);


  const handleStartupChange = (event) => {
    setSelectedStartup(event.target.value);
  };

  const handleVisualizationTypeChange = (event) => {
    setSelectedVisualizationType(event.target.value);
  };


  const handleDataPointChange = (event) => {
    const { target: { value } } = event;
    setSelectedDataPoints(
      // On autofill we get a stringified value.
      typeof value === "string" ? value.split(",") : value
    );
  };

  const handleTimeRangeChange = (event, newValue) => {
    setTimeRange(newValue);
  };

  const renderVisualization = () => {
    if (selectedVisualizationType === 'LineChart') {
      return (
        <LineChart width={800} height={400} data={visualizationData}>
          <XAxis dataKey="date" />
          <YAxis />
          <CartesianGrid stroke="#f5f5f5" />
          <Tooltip />
          <Legend />
          {selectedDataPoints.map((dataPoint) => (
            <Line type="monotone" dataKey={dataPoint} stroke="#4285F4" key={dataPoint} />
          ))}
        </LineChart>
      );
    }
    // 他のグラフタイプ (棒グラフなど) に対する処理も追加

    return null;
  };


  return (
    <div>
      <Navigation />
      <Grid container spacing={3} mt={2}>
        <Grid item xs={12}>
          <Typography variant="h4" align="center">
            Visualization
          </Typography>
        </Grid>
        {/* スタートアップ選択 */}
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

        {/* 可視化タイプ選択 */}
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel id="visualization-type-select-label">Visualization Type</InputLabel>
            <Select
              labelId="visualization-type-select-label"
              value={selectedVisualizationType}
              label="Visualization Type"
              onChange={handleVisualizationTypeChange}
            >
              {availableVisualizationTypes.map((type) => (
                <MenuItem key={type} value={type}>
                  {type}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        {/* データポイント選択 */}
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel id="data-point-select-label">Data Points</InputLabel>
            <Select
              labelId="data-point-select-label"
              multiple
              value={selectedDataPoints}
              onChange={handleDataPointChange}
              input={<OutlinedInput label="Data Points" />}
              renderValue={(selected) => selected.join(', ')}
              MenuProps={MenuProps}
            >
              {availableDataPoints.map((dataPoint) => (
                <MenuItem key={dataPoint} value={dataPoint}>
                  <Checkbox checked={selectedDataPoints.indexOf(dataPoint) > -1} />
                  <ListItemText primary={dataPoint} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* 時間範囲選択スライダー */}
        <Grid item xs={12} md={8}>
          <Typography id="time-range-slider" gutterBottom>
            Time Range ({timeRange[0]} - {timeRange[1]})
          </Typography>
          <Slider
            value={timeRange}
            onChange={handleTimeRangeChange}
            valueLabelDisplay="auto"
            min={2015}
            max={2025}
          />
        </Grid>
        {/* 可視化結果表示エリア */}
        <Grid item xs={12}>
          {renderVisualization()}
        </Grid>
      </Grid>
    </div>
  );
};


export default Visualization;