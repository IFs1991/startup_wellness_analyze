import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { FileText, Download, Printer, Plus } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

const reportTypes = [
  { id: 'pdf', label: 'PDF' },
  { id: 'excel', label: 'Excel' },
  { id: 'csv', label: 'CSV' },
];

const ReportsPage: React.FC = () => {
  // ダミーのレポートデータ
  const reports = [
    { id: '1', title: '四半期業績レポート', date: '2025-03-01', type: '定期レポート' },
    { id: '2', title: 'チーム健康度分析', date: '2025-02-25', type: '特別分析' },
    { id: '3', title: '投資候補企業レポート', date: '2025-02-15', type: '投資分析' },
    { id: '4', title: '業界トレンド分析', date: '2025-02-10', type: '市場調査' },
  ];

  return (
    <div className="container mx-auto">
      <div className="my-8">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold">
            レポート
          </h1>
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            新規レポート作成
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {reports.map((report) => (
            <Card
              key={report.id}
              className="transition-all hover:translate-y-[-3px] hover:shadow-md"
            >
              <CardHeader>
                <CardTitle>{report.title}</CardTitle>
                <p className="text-sm text-muted-foreground">
                  作成日: {new Date(report.date).toLocaleDateString('ja-JP')}
                </p>
              </CardHeader>
              <CardContent>
                <p className="text-sm mb-2">
                  種類: {report.type}
                </p>
              </CardContent>
              <CardFooter className="flex gap-2">
                <Button variant="outline" size="sm">
                  表示
                </Button>
                <Button variant="outline" size="sm">
                  <Download className="mr-2 h-4 w-4" />
                  ダウンロード
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ReportsPage;