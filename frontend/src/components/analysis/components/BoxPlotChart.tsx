import React from 'react';
import {
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceArea,
} from 'recharts';
import { Paper, Typography } from '@mui/material';

interface BoxPlotChartProps {
  data: {
    min: number;
    q1: number;
    median: number;
    q3: number;
    max: number;
  };
  height?: number;
  width?: string | number;
}

export const BoxPlotChart: React.FC<BoxPlotChartProps> = ({
  data,
  height = 100,
  width = '100%',
}) => {
  const { min, q1, median, q3, max } = data;

  return (
    <ResponsiveContainer width={width} height={height}>
      <BarChart
        layout="horizontal"
        data={[{ name: '残差', value: median }]}
        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          type="number"
          domain={[min, max]}
          label={{ value: '残差値', position: 'bottom' }}
        />
        <YAxis dataKey="name" type="category" />
        <RechartsTooltip
          content={({ active }) => {
            if (active) {
              return (
                <Paper sx={{ p: 1 }}>
                  <Typography variant="body2">
                    最小値: {min.toFixed(2)}
                  </Typography>
                  <Typography variant="body2">
                    第1四分位数: {q1.toFixed(2)}
                  </Typography>
                  <Typography variant="body2">
                    中央値: {median.toFixed(2)}
                  </Typography>
                  <Typography variant="body2">
                    第3四分位数: {q3.toFixed(2)}
                  </Typography>
                  <Typography variant="body2">
                    最大値: {max.toFixed(2)}
                  </Typography>
                </Paper>
              );
            }
            return null;
          }}
        />
        {/* 箱の部分 */}
        <ReferenceArea
          x1={q1}
          x2={q3}
          fill="#8884d8"
          fillOpacity={0.3}
        />
        {/* 中央線 */}
        <ReferenceArea
          x1={median}
          x2={median}
          fill="#000"
          fillOpacity={1}
        />
        {/* ひげの部分 */}
        <ReferenceArea
          x1={min}
          x2={max}
          fill="none"
          stroke="#000"
          strokeWidth={1}
        />
      </BarChart>
    </ResponsiveContainer>
  );
};