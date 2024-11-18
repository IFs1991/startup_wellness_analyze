import React, { useEffect, useState } from "react";
import { Grid, Typography } from "@mui/material";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import DataService from "../services/DataService";

const Dashboard = () => {
  const [startupData, setStartupData] = useState([]);

  useEffect(() => {
    const fetchStartupData = async () => {
      try {
        const data = await DataService.getStartupData();
        setStartupData(data);
      } catch (error) {
        console.error("Failed to fetch startup data:", error);
      }
    };

    fetchStartupData();
  }, []);

  return (
    <Grid container spacing={3}>
      {/* スタートアップ毎のデータを表示 */}
      {startupData.map((startup) => (
        <Grid item xs={12} md={6} key={startup.id}>
          <Typography variant="h6" gutterBottom>
            {startup.name}
          </Typography>
          {/* VASデータの可視化 */}
          <LineChart width={500} height={300} data={startup.vasData}>
            <XAxis dataKey="timestamp" />
            <YAxis />
            <CartesianGrid stroke="#f5f5f5" />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="physical_symptoms" name="Physical Symptoms" stroke="#4285F4" />
            {/* 他のVASデータ項目を追加 */}
          </LineChart>

          {/* 財務データの表示 */}
          <Typography variant="body1" mt={2}>
            Revenue: {startup.financialData.revenue}
          </Typography>
          {/* 他の財務データ項目を追加 */}
        </Grid>
      ))}
    </Grid>
  );
};

export default Dashboard;