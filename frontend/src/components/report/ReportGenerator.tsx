import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, FileText, Download, Send } from 'lucide-react';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';

interface ReportGeneratorProps {
  companyId: string;
  companyName: string;
}

// レポート項目のオプション
const reportOptions = [
  { id: 'overview', label: '企業概要' },
  { id: 'wellness', label: 'ウェルネス状況' },
  { id: 'financial', label: '財務状況との関連' },
  { id: 'recommendations', label: '改善提案' },
  { id: 'forecast', label: '将来予測' }
];

// レポートフォーマットのオプション
const formatOptions = [
  { id: 'pdf', label: 'PDF' },
  { id: 'excel', label: 'Excel' },
  { id: 'ppt', label: 'PowerPoint' }
];

export function ReportGenerator({ companyId, companyName }: ReportGeneratorProps) {
  const [selectedSections, setSelectedSections] = useState<string[]>(['overview', 'wellness']);
  const [selectedFormat, setSelectedFormat] = useState('pdf');
  const [customPrompt, setCustomPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedReport, setGeneratedReport] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('configure');

  const toggleSection = (sectionId: string) => {
    setSelectedSections(prev =>
      prev.includes(sectionId)
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    );
  };

  const generateReport = async () => {
    setIsGenerating(true);
    try {
      // 実際のAPIが実装されるまではモックのレポート生成をシミュレーション
      await new Promise(resolve => setTimeout(resolve, 2000));

      // モックのレポートテキスト
      const mockReport = `
# ${companyName} ウェルネス分析レポート

## 企業概要
${companyName}は最新のテクノロジーを活用し、持続可能な事業成長を目指している企業です。従業員数は約200名で、テクノロジー業界において中堅企業として位置づけられています。

## ウェルネス状況
現在のウェルネススコアは82/100となっており、業界平均の74/100を上回っています。特に以下の分野で高いスコアを記録しています：
- エンゲージメント：85/100
- ワークライフバランス：81/100
- 職場環境：88/100

## 財務状況との関連
${selectedSections.includes('financial') ? `
ウェルネススコアと財務指標の間には明確な相関関係が見られます：
- 過去3年間のウェルネススコア上昇と売上成長率には0.85の正の相関
- 従業員満足度の向上により離職率は業界平均と比較して30%低い
- 生産性は業界平均を12%上回り、利益率の向上に貢献` : ''}

${selectedSections.includes('recommendations') ? `
## 改善提案
1. リモートワークポリシーのさらなる拡充
2. メンタルヘルスサポートプログラムの導入
3. 定期的なエンゲージメント調査の実施と改善サイクルの確立` : ''}

${selectedSections.includes('forecast') ? `
## 将来予測
現在のトレンドが継続した場合：
- 2年以内にウェルネススコアは85-90に到達する見込み
- 離職率はさらに10%低下する可能性
- 生産性は追加で8%向上し、利益率の改善に貢献` : ''}

## カスタムレポート情報
${customPrompt ? `追加リクエスト: ${customPrompt}` : '特になし'}

レポート生成日: ${new Date().toLocaleDateString('ja-JP')}
`;

      setGeneratedReport(mockReport);
      setActiveTab('preview');
    } catch (error) {
      console.error('レポート生成中にエラーが発生しました:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadReport = () => {
    if (!generatedReport) return;

    const blob = new Blob([generatedReport], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${companyName}_ウェルネス分析レポート.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const sendReport = () => {
    // メール送信をシミュレーション
    alert(`${companyName}のレポートをメールで送信しました。`);
  };

  return (
    <Card className="p-4">
      <h3 className="text-lg font-medium mb-2">レポート生成</h3>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="mt-4">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="configure">設定</TabsTrigger>
          <TabsTrigger value="preview" disabled={!generatedReport}>
            プレビュー {generatedReport && <FileText className="ml-2 h-4 w-4" />}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="configure" className="mt-4">
          <div className="space-y-6">
            <div>
              <h4 className="text-sm font-medium mb-3">レポートに含める項目</h4>
              <div className="grid gap-3">
                {reportOptions.map(option => (
                  <div key={option.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={option.id}
                      checked={selectedSections.includes(option.id)}
                      onCheckedChange={() => toggleSection(option.id)}
                    />
                    <Label htmlFor={option.id}>{option.label}</Label>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <Label htmlFor="report-format" className="text-sm font-medium">レポート形式</Label>
              <Select
                value={selectedFormat}
                onValueChange={setSelectedFormat}
              >
                <SelectTrigger className="w-full mt-1">
                  <SelectValue placeholder="形式を選択" />
                </SelectTrigger>
                <SelectContent>
                  {formatOptions.map(format => (
                    <SelectItem key={format.id} value={format.id}>
                      {format.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="custom-prompt" className="text-sm font-medium">追加リクエスト</Label>
              <Textarea
                id="custom-prompt"
                placeholder="レポートに含めたい特定の情報や観点があれば入力してください"
                className="mt-1"
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
              />
            </div>

            <Button
              onClick={generateReport}
              disabled={isGenerating || selectedSections.length === 0}
              className="w-full"
            >
              {isGenerating && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              レポートを生成
            </Button>
          </div>
        </TabsContent>

        <TabsContent value="preview" className="mt-4">
          {generatedReport && (
            <div className="space-y-4">
              <div className="bg-muted p-4 rounded-md whitespace-pre-wrap text-sm overflow-auto max-h-[400px]">
                {generatedReport}
              </div>

              <div className="flex gap-2">
                <Button onClick={downloadReport} className="flex-1">
                  <Download className="mr-2 h-4 w-4" />
                  ダウンロード
                </Button>
                <Button variant="outline" onClick={sendReport} className="flex-1">
                  <Send className="mr-2 h-4 w-4" />
                  メールで送信
                </Button>
              </div>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </Card>
  );
}