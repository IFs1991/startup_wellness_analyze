import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { useChartTheme } from '@/hooks/useChartTheme';
import { ChartContainer } from './ChartContainer';

interface DataPoint {
  name: string;
  value: number;
}

interface MonthlyTrendChartProps {
  data: DataPoint[];
}

export function MonthlyTrendChart({ data }: MonthlyTrendChartProps) {
  const theme = useChartTheme();
  
  return (
    <ChartContainer>
      <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid 
          strokeDasharray="3 3" 
          stroke={theme.grid.stroke}
        />
        <XAxis 
          dataKey="name"
          padding={{ left: 20, right: 20 }}
          stroke={theme.axis.stroke}
          fontSize={theme.axis.fontSize}
          tickLine={{ stroke: theme.axis.stroke }}
          axisLine={{ stroke: theme.axis.stroke }}
        />
        <YAxis
          width={60}
          stroke={theme.axis.stroke}
          fontSize={theme.axis.fontSize}
          tickLine={{ stroke: theme.axis.stroke }}
          axisLine={{ stroke: theme.axis.stroke }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: theme.tooltip.background,
            border: `1px solid ${theme.tooltip.border}`,
            borderRadius: '8px',
            color: theme.tooltip.color
          }}
        />
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke="#4285F4" 
          strokeWidth={2}
          dot={{ fill: '#4285F4', strokeWidth: 0 }}
          activeDot={{ r: 6, strokeWidth: 0 }}
        />
      </LineChart>
    </ChartContainer>
  );
}