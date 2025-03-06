import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Loader2, Save, History, Info, DollarSign, Calendar, Clock } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { companiesApi } from '@/lib/api/companies';
import type { Company, CompanyDetail, EditHistoryEntry } from '@/types/company';

interface EditCompanyDialogProps {
  company: CompanyDetail;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onCompanyUpdated: (updatedCompany: CompanyDetail) => void;
}

const stages = [
  { value: 'pre-seed', label: 'プレシード' },
  { value: 'seed', label: 'シード' },
  { value: 'early', label: 'アーリー' },
  { value: 'series-a', label: 'シリーズA' },
  { value: 'series-b', label: 'シリーズB' },
  { value: 'series-c', label: 'シリーズC' },
  { value: 'series-d', label: 'シリーズD以降' },
  { value: 'pre-ipo', label: 'プレIPO' },
];

export function EditCompanyDialog({ company, isOpen, onOpenChange, onCompanyUpdated }: EditCompanyDialogProps) {
  const [formData, setFormData] = useState<CompanyDetail>({ ...company });
  const [editHistory, setEditHistory] = useState<EditHistoryEntry[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [activeTab, setActiveTab] = useState('edit');
  const { toast } = useToast();

  // 変更されたフィールドを追跡
  const [changedFields, setChangedFields] = useState<string[]>([]);

  // 会社データが変更されたときにフォームデータを更新
  useEffect(() => {
    setFormData({ ...company });
  }, [company]);

  // タブが「履歴」に切り替わったときに履歴データを取得
  useEffect(() => {
    if (activeTab === 'history') {
      fetchEditHistory();
    }
  }, [activeTab]);

  // 編集履歴を取得
  const fetchEditHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const history = await companiesApi.getEditHistory(company.id);
      setEditHistory(history);
    } catch (error) {
      console.error('編集履歴の取得に失敗しました:', error);
      toast({
        title: 'エラー',
        description: '編集履歴の取得に失敗しました。',
        variant: 'destructive',
      });
    } finally {
      setIsLoadingHistory(false);
    }
  };

  // フォームフィールドの変更を処理
  const handleFieldChange = (field: keyof CompanyDetail, value: CompanyDetail[keyof CompanyDetail]) => {
    // 値が変更された場合のみ変更フィールドに追加
    if (company[field] !== value && !changedFields.includes(field as string)) {
      setChangedFields([...changedFields, field as string]);
    } else if (company[field] === value && changedFields.includes(field as string)) {
      // 値が元に戻った場合は変更フィールドから削除
      setChangedFields(changedFields.filter(f => f !== field));
    }

    setFormData({
      ...formData,
      [field]: value
    });
  };

  // フォーム送信処理
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (changedFields.length === 0) {
      toast({
        title: '変更なし',
        description: '変更された項目がありません。',
      });
      return;
    }

    try {
      setIsSubmitting(true);

      // 更新するデータを準備
      const updateData: Partial<CompanyDetail> = {};
      changedFields.forEach(field => {
        // 型安全な方法でフィールドを設定
        if (field in formData) {
          (updateData as any)[field] = formData[field as keyof typeof formData];
        }
      });

      // APIを使用して企業情報を更新
      const updatedCompany = await companiesApi.updateCompany(
        company.id,
        updateData,
        changedFields
      );

      onCompanyUpdated(updatedCompany);

      toast({
        title: '更新完了',
        description: '企業情報が更新されました。',
      });

      setChangedFields([]);
      onOpenChange(false);
    } catch (error) {
      toast({
        title: 'エラー',
        description: '企業情報の更新に失敗しました。',
        variant: 'destructive',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  // フィールドが変更されたかどうかを確認
  const isFieldChanged = (field: string) => changedFields.includes(field);

  // 編集履歴の表示用に日本語フィールド名に変換
  const getFieldDisplayName = (field: string): string => {
    const fieldMap: Record<string, string> = {
      name: '企業名',
      industry: '業界',
      stage: 'ステージ',
      employees: '従業員数',
      foundedYear: '設立年',
      location: '所在地',
      description: '説明',
      totalFunding: '累計調達額'
    };
    return fieldMap[field] || field;
  };

  // フィールドアイコンを取得
  const getFieldIcon = (field: string) => {
    const iconMap: Record<string, JSX.Element> = {
      foundedYear: <Calendar className="h-4 w-4" />,
      totalFunding: <DollarSign className="h-4 w-4" />
    };
    return iconMap[field] || <Info className="h-4 w-4" />;
  };

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>企業情報編集</DialogTitle>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
          <TabsList className="grid grid-cols-2 mb-4">
            <TabsTrigger value="edit">
              <Info className="h-4 w-4 mr-2" />
              編集
            </TabsTrigger>
            <TabsTrigger value="history">
              <History className="h-4 w-4 mr-2" />
              編集履歴
            </TabsTrigger>
          </TabsList>

          <TabsContent value="edit" className="h-full overflow-hidden flex flex-col">
            <ScrollArea className="flex-1 pr-4">
              <form id="edit-company-form" onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name" className="flex items-center">
                    企業名 *
                    {isFieldChanged('name') && <span className="ml-2 text-xs text-primary">(変更済)</span>}
                  </Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => handleFieldChange('name', e.target.value)}
                    required
                    className={isFieldChanged('name') ? 'border-primary' : ''}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="industry" className="flex items-center">
                      業界 *
                      {isFieldChanged('industry') && <span className="ml-2 text-xs text-primary">(変更済)</span>}
                    </Label>
                    <Input
                      id="industry"
                      value={formData.industry}
                      onChange={(e) => handleFieldChange('industry', e.target.value)}
                      placeholder="例: SaaS, フィンテック"
                      required
                      className={isFieldChanged('industry') ? 'border-primary' : ''}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="stage" className="flex items-center">
                      ステージ *
                      {isFieldChanged('stage') && <span className="ml-2 text-xs text-primary">(変更済)</span>}
                    </Label>
                    <Select
                      value={formData.stage}
                      onValueChange={(value) => handleFieldChange('stage', value)}
                      required
                    >
                      <SelectTrigger className={isFieldChanged('stage') ? 'border-primary' : ''}>
                        <SelectValue placeholder="ステージを選択" />
                      </SelectTrigger>
                      <SelectContent>
                        {stages.map((stage) => (
                          <SelectItem key={stage.value} value={stage.value}>
                            {stage.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="employees" className="flex items-center">
                      従業員数
                      {isFieldChanged('employees') && <span className="ml-2 text-xs text-primary">(変更済)</span>}
                    </Label>
                    <Input
                      id="employees"
                      type="number"
                      min="1"
                      value={formData.employees}
                      onChange={(e) => handleFieldChange('employees', parseInt(e.target.value))}
                      className={isFieldChanged('employees') ? 'border-primary' : ''}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="foundedYear" className="flex items-center">
                      設立年
                      {isFieldChanged('foundedYear') && <span className="ml-2 text-xs text-primary">(変更済)</span>}
                    </Label>
                    <Input
                      id="foundedYear"
                      type="number"
                      min="1900"
                      max={new Date().getFullYear()}
                      value={formData.foundedYear || ''}
                      onChange={(e) => handleFieldChange('foundedYear', parseInt(e.target.value))}
                      className={isFieldChanged('foundedYear') ? 'border-primary' : ''}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="location" className="flex items-center">
                      所在地
                      {isFieldChanged('location') && <span className="ml-2 text-xs text-primary">(変更済)</span>}
                    </Label>
                    <Input
                      id="location"
                      value={formData.location || ''}
                      onChange={(e) => handleFieldChange('location', e.target.value)}
                      className={isFieldChanged('location') ? 'border-primary' : ''}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="totalFunding" className="flex items-center">
                      累計調達額
                      {isFieldChanged('totalFunding') && <span className="ml-2 text-xs text-primary">(変更済)</span>}
                    </Label>
                    <div className="relative">
                      <DollarSign className="absolute left-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        id="totalFunding"
                        value={formData.totalFunding || ''}
                        onChange={(e) => handleFieldChange('totalFunding', e.target.value)}
                        className={`pl-8 ${isFieldChanged('totalFunding') ? 'border-primary' : ''}`}
                        placeholder="例: 5.2億円"
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="description" className="flex items-center">
                    説明
                    {isFieldChanged('description') && <span className="ml-2 text-xs text-primary">(変更済)</span>}
                  </Label>
                  <Textarea
                    id="description"
                    rows={3}
                    value={formData.description}
                    onChange={(e) => handleFieldChange('description', e.target.value)}
                    className={isFieldChanged('description') ? 'border-primary' : ''}
                  />
                </div>
              </form>
            </ScrollArea>

            <div className="flex justify-end space-x-2 pt-4 mt-4 border-t">
              <Button variant="outline" type="button" onClick={() => onOpenChange(false)}>
                キャンセル
              </Button>
              <Button
                type="submit"
                form="edit-company-form"
                disabled={isSubmitting || changedFields.length === 0}
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    更新中...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    保存
                  </>
                )}
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="history" className="h-full overflow-hidden flex flex-col">
            <ScrollArea className="flex-1">
              {isLoadingHistory ? (
                <div className="flex justify-center items-center h-64">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : editHistory.length > 0 ? (
                <div className="space-y-4">
                  {editHistory.map((entry) => (
                    <Card key={entry.id}>
                      <CardHeader className="py-3">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                          {getFieldIcon(entry.field)}
                          {getFieldDisplayName(entry.field)}の変更
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="py-2">
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground flex items-center gap-1">
                              <Clock className="h-3.5 w-3.5" />
                              日時:
                            </span>
                            <span>{entry.date}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">変更者:</span>
                            <span>{entry.editedBy || 'システム'}</span>
                          </div>
                          <div className="pt-2 border-t">
                            <div className="flex mb-1">
                              <span className="text-muted-foreground w-1/2">変更前:</span>
                              <span className="text-muted-foreground w-1/2">変更後:</span>
                            </div>
                            <div className="flex">
                              <div className="bg-muted p-2 rounded-l-md w-1/2 overflow-hidden text-ellipsis">
                                {entry.oldValue || '(未設定)'}
                              </div>
                              <div className="bg-primary/10 p-2 rounded-r-md w-1/2 overflow-hidden text-ellipsis font-medium">
                                {entry.newValue || '(未設定)'}
                              </div>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  編集履歴はありません
                </div>
              )}
            </ScrollArea>

            <div className="flex justify-end space-x-2 pt-4 mt-4 border-t">
              <Button onClick={() => setActiveTab('edit')}>
                編集画面に戻る
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}