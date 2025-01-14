import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useChartTheme } from '@/hooks/useChartTheme';

interface FinancialData {
  date: string;
  revenue: number;
  profit: number;
  assets: number;
  liabilities: number;
}

interface FinancialMetricsProps {
  data: FinancialData[];
}

export function FinancialMetrics({ data }: FinancialMetricsProps) {
  const theme = useChartTheme();

  return (
    <Card className="p-6">
      <h2 className="text-xl font-semibold mb-4">財務指標</h2>
      <Tabs defaultValue="pl">
        <TabsList>
          <TabsTrigger value="pl">損益計算書</TabsTrigger>
          <TabsTrigger value="bs">貸借対照表</TabsTrigger>
        </TabsList>

        <TabsContent value="pl" className="h-[400px] mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.grid.stroke} />
              <XAxis dataKey="date" stroke={theme.axis.stroke} />
              <YAxis stroke={theme.axis.stroke} />
              <Tooltip
                contentStyle={{
                  backgroundColor: theme.tooltip.background,
                  border: `1px solid ${theme.tooltip.border}`,
                }}
              />
              <Line type="monotone" dataKey="revenue" name="売上高" stroke="#4285F4" />
              <Line type="monotone" dataKey="profit" name="利益" stroke="#34A853" />
            </LineChart>
          </ResponsiveContainer>
        </TabsContent>

        <TabsContent value="bs" className="h-[400px] mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.grid.stroke} />
              <XAxis dataKey="date" stroke={theme.axis.stroke} />
              <YAxis stroke={theme.axis.stroke} />
              <Tooltip
                contentStyle={{
                  backgroundColor: theme.tooltip.background,
                  border: `1px solid ${theme.tooltip.border}`,
                }}
              />
              <Line type="monotone" dataKey="assets" name="資産" stroke="#4285F4" />
              <Line type="monotone" dataKey="liabilities" name="負債" stroke="#EA4335" />
            </LineChart>
          </ResponsiveContainer>
        </TabsContent>
      </Tabs>
    </Card>
  );
}