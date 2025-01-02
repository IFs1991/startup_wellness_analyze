import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useChartTheme } from '@/hooks/useChartTheme';
import { Card } from '@/components/ui/card';
import { Info } from 'lucide-react';
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface DataPoint {
  company: string;
  wellnessScore: number;
  growthRate: number;
  industry: string;
}

interface WellnessScoreChartProps {
  data: DataPoint[];
}

export function WellnessScoreChart({ data }: WellnessScoreChartProps) {
  const theme = useChartTheme();

  return (
    <Card className="p-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold">ウェルネススコアと売上成長率の相関</h3>
              <TooltipProvider>
                <UITooltip>
                  <TooltipTrigger>
                    <Info className="h-4 w-4 text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">
                      ウェルネススコアは従業員の健康状態、満足度、
                      エンゲージメントを総合的に評価した指標です。
                      スコアが高いほど、従業員の健康経営が進んでいることを示します。
                    </p>
                  </TooltipContent>
                </UITooltip>
              </TooltipProvider>
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              ウェルネススコアが高い企業ほど、売上成長率が高い傾向にあります
            </p>
          </div>
        </div>
        
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.grid.stroke} />
              <XAxis
                type="number"
                dataKey="wellnessScore"
                name="ウェルネススコア"
                domain={[0, 100]}
                stroke={theme.axis.stroke}
                fontSize={theme.axis.fontSize}
                tickLine={{ stroke: theme.axis.stroke }}
                axisLine={{ stroke: theme.axis.stroke }}
                label={{ 
                  value: 'ウェルネススコア',
                  position: 'bottom',
                  offset: 0,
                  style: { fill: theme.axis.stroke }
                }}
              />
              <YAxis
                type="number"
                dataKey="growthRate"
                name="売上成長率"
                unit="%"
                stroke={theme.axis.stroke}
                fontSize={theme.axis.fontSize}
                tickLine={{ stroke: theme.axis.stroke }}
                axisLine={{ stroke: theme.axis.stroke }}
                label={{ 
                  value: '売上成長率 (%)',
                  angle: -90,
                  position: 'left',
                  offset: 0,
                  style: { fill: theme.axis.stroke }
                }}
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{
                  backgroundColor: theme.tooltip.background,
                  border: `1px solid ${theme.tooltip.border}`,
                  borderRadius: '8px',
                  color: theme.tooltip.color
                }}
                formatter={(value: number, name: string) => {
                  if (name === 'ウェルネススコア') return [value.toFixed(1), name];
                  if (name === '売上成長率') return [`${value.toFixed(1)}%`, name];
                  return [value, name];
                }}
              />
              <Legend />
              <Scatter
                name="企業"
                data={data}
                fill="#4285F4"
                fillOpacity={0.6}
                stroke="#4285F4"
                strokeWidth={1}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Card>
  );
}