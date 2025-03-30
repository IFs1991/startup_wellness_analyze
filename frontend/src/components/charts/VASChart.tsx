import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface VASData {
  timestamp: string;
  pain_level: number;
  stress_level: number;
  sleep_quality: number;
}

interface VASChartProps {
  data: VASData[];
}

export const VASChart: React.FC<VASChartProps> = ({ data }) => {
  const chartData = {
    labels: data.map(d => new Date(d.timestamp).toLocaleDateString('ja-JP')),
    datasets: [
      {
        label: '痛みレベル',
        data: data.map(d => d.pain_level),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
      {
        label: 'ストレスレベル',
        data: data.map(d => d.stress_level),
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      },
      {
        label: '睡眠の質',
        data: data.map(d => d.sleep_quality),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'VASスケール推移',
      },
    },
    scales: {
      y: {
        min: 0,
        max: 100,
      },
    },
  };

  return (
    <div className="w-full h-[400px]">
      <Line data={chartData} options={options} />
    </div>
  );
};