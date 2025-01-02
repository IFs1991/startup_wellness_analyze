import { useState } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { Progress } from '@/components/ui/progress';

interface FinancialDataUploadProps {
  companyId: string;
  onUploadComplete: () => void;
}

export function FinancialDataUpload({ companyId, onUploadComplete }: FinancialDataUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [documentType, setDocumentType] = useState('');
  const { toast } = useToast();

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !documentType) return;

    setUploading(true);
    setProgress(0);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', documentType);

    try {
      // Simulate upload progress
      const interval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 500);

      // TODO: Implement actual file upload
      await new Promise(resolve => setTimeout(resolve, 2000));
      clearInterval(interval);
      setProgress(100);

      toast({
        title: 'アップロード完了',
        description: `${file.name}のアップロードが完了しました。`,
      });
      
      onUploadComplete();
    } catch (error) {
      toast({
        title: 'エラー',
        description: 'ファイルのアップロードに失敗しました。',
        variant: 'destructive',
      });
    } finally {
      setUploading(false);
      setProgress(0);
      setDocumentType('');
    }
  };

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">財務データアップロード</h3>
      
      <div className="space-y-4">
        <Select value={documentType} onValueChange={setDocumentType}>
          <SelectTrigger>
            <SelectValue placeholder="ドキュメントタイプを選択" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="pl">損益計算書 (PL)</SelectItem>
            <SelectItem value="bs">貸借対照表 (BS)</SelectItem>
            <SelectItem value="cf">キャッシュフロー計算書</SelectItem>
            <SelectItem value="other">その他財務資料</SelectItem>
          </SelectContent>
        </Select>

        <div className="flex items-center justify-center border-2 border-dashed border-border rounded-lg p-6">
          <div className="text-center">
            <Upload className="mx-auto h-12 w-12 text-muted-foreground" />
            <div className="mt-4">
              <Button disabled={!documentType || uploading} className="relative">
                <input
                  type="file"
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  accept=".csv,.xlsx,.pdf"
                  onChange={handleFileUpload}
                />
                <FileText className="h-4 w-4 mr-2" />
                ファイルを選択
              </Button>
            </div>
            <p className="text-sm text-muted-foreground mt-2">
              CSV, Excel, PDFファイルをアップロード
            </p>
          </div>
        </div>

        {uploading && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>アップロード中...</span>
              <span>{progress}%</span>
            </div>
            <Progress value={progress} />
          </div>
        )}

        <div className="flex items-start space-x-2 text-sm text-muted-foreground">
          <AlertCircle className="h-4 w-4 mt-0.5" />
          <p>アップロードされたデータは自動的に分析され、グラフやレポートに反映されます</p>
        </div>
      </div>
    </Card>
  );
}