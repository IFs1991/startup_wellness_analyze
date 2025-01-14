import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useChartTheme } from '@/hooks/useChartTheme';
import { Card } from '@/components/ui/card';

interface WellnessChartProps {
  trends: Array<{
    date: string;
    score: number;
  }>;
}

export function WellnessChart({ trends }: WellnessChartProps) {
  const theme = useChartTheme();

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">ウェルネススコアの推移</h3>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={trends}>
            <CartesianGrid strokeDasharray="3 3" stroke={theme.grid.stroke} />
            <XAxis
              dataKey="date"
              stroke={theme.axis.stroke}
              fontSize={theme.axis.fontSize}
            />
            <YAxis
              stroke={theme.axis.stroke}
              fontSize={theme.axis.fontSize}
              domain={[0, 100]}
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
              dataKey="score"
              stroke="#4285F4"
              strokeWidth={2}
              dot={{ fill: '#4285F4', strokeWidth: 0 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}