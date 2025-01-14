import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { FileText, Download } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';

interface ReportGeneratorProps {
  companyId: string;
}

export function ReportGenerator({ companyId }: ReportGeneratorProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const { toast } = useToast();

  const handleGenerate = async () => {
    setIsGenerating(true);
    setProgress(0);

    try {
      // Simulate report generation progress
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            return 100;
          }
          return prev + 10;
        });
      }, 500);

      // TODO: Implement actual report generation API call
      await fetch(`/api/reports/generate/${companyId}`, {
        method: 'POST',
      });

      toast({
        title: 'レポート生成完了',
        description: 'レポートのダウンロードが可能になりました',
      });
    } catch (error) {
      console.error('Failed to generate report:', error);
      toast({
        title: 'エラー',
        description: 'レポートの生成に失敗しました',
        variant: 'destructive',
      });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Card className="p-6">
      <h2 className="text-xl font-semibold mb-4">レポート生成</h2>
      
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <Select defaultValue="comprehensive">
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="レポートタイプを選択" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="comprehensive">総合分析レポート</SelectItem>
              <SelectItem value="wellness">健康スコア詳細</SelectItem>
              <SelectItem value="financial">財務分析レポート</SelectItem>
            </SelectContent>
          </Select>

          <Button onClick={handleGenerate} disabled={isGenerating}>
            <FileText className="h-4 w-4 mr-2" />
            レポート生成
          </Button>
        </div>

        {isGenerating && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>生成中...</span>
              <span>{progress}%</span>
            </div>
            <Progress value={progress} />
          </div>
        )}

        {progress === 100 && (
          <Button variant="outline" className="w-full">
            <Download className="h-4 w-4 mr-2" />
            レポートをダウンロード
          </Button>
        )}
      </div>
    </Card>
  );
}