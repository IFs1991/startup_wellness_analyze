import React, { useState, useEffect } from 'react';
import { Grid, Typography, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import VisualizationService from '../services/VisualizationService';

const DataVisualization = () => {
  const [chartData, setChartData] = useState([]);
  const [selectedStartup, setSelectedStartup] = useState('');
  const [availableStartups, setAvailableStartups] = useState([]);

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        const data = await VisualizationService.getChartData();
        setChartData(data);
        setAvailableStartups(data.map((item) => item.startup).filter((startup, index, self) => self.indexOf(startup) === index));
        setSelectedStartup(availableStartups[0] || ''); // デフォルトで最初のスタートアップを選択
      } catch (error) {
        console.error('Failed to fetch chart data:', error);
      }
    };

    fetchChartData();
  }, []);

  const handleStartupChange = (event) => {
    setSelectedStartup(event.target.value);
  };

  const filteredChartData = chartData.filter((item) => item.startup === selectedStartup);

  return (
    <Grid container spacing={2} justifyContent="center">
      <Grid item xs={12}>
        <Typography variant="h6" align="center" gutterBottom>
          Data Visualization
        </Typography>
      </Grid>
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel id="startup-select-label">Startup</InputLabel>
          <Select labelId="startup-select-label" value={selectedStartup} label="Startup" onChange={handleStartupChange}>
            {availableStartups.map((startup) => (
              <MenuItem key={startup} value={startup}>
                {startup}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={12} md={8}>
        <LineChart width={600} height={300} data={filteredChartData}>
          <XAxis dataKey="date" />
          <YAxis />
          <CartesianGrid stroke="#f5f5f5" />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="value" stroke="#4285F4" />
        </LineChart>
      </Grid>
    </Grid>
  );
};

export default DataVisualization;