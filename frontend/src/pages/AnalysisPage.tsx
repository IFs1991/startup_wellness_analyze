import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MonthlyTrendChart } from '@/components/charts/MonthlyTrendChart';
import { DataTable } from '@/components/tables/DataTable';

const data = [
  { name: '1月', value: 400 },
  { name: '2月', value: 300 },
  { name: '3月', value: 600 },
  { name: '4月', value: 800 },
  { name: '5月', value: 500 },
];

export function AnalysisPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-[#212121]">分析ダッシュボード</h1>
      
      <Tabs defaultValue="chart" className="w-full">
        <TabsList>
          <TabsTrigger value="chart">チャート</TabsTrigger>
          <TabsTrigger value="data">データ</TabsTrigger>
        </TabsList>
        
        <TabsContent value="chart">
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">月次トレンド</h2>
            <MonthlyTrendChart data={data} />
          </Card>
        </TabsContent>
        
        <TabsContent value="data">
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">データテーブル</h2>
            <DataTable data={data} />
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}