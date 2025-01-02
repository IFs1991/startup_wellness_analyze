import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { useChartTheme } from '@/hooks/useChartTheme';

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
    <div className="h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart 
          data={data} 
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke={theme.grid.stroke}
          />
          <XAxis 
            dataKey="name"
            stroke={theme.axis.stroke}
            fontSize={theme.axis.fontSize}
            tickLine={{ stroke: theme.axis.stroke }}
            axisLine={{ stroke: theme.axis.stroke }}
            padding={{ left: 20, right: 20 }}
            allowDataOverflow={false}
            scale="point"
            type="category"
          />
          <YAxis
            stroke={theme.axis.stroke}
            fontSize={theme.axis.fontSize}
            tickLine={{ stroke: theme.axis.stroke }}
            axisLine={{ stroke: theme.axis.stroke }}
            width={60}
            allowDataOverflow={false}
            scale="linear"
            type="number"
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
      </ResponsiveContainer>
    </div>
  );
}