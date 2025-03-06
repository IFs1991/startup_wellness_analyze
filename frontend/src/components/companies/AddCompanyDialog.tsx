import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { PlusCircle, Loader2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { companiesApi } from '@/lib/api/companies';

interface CompanyFormData {
  name: string;
  industry: string;
  stage: string;
  employees: number;
  foundedYear: number;
  location: string;
  ceo: string;
  investment: number;
}

const initialFormData: CompanyFormData = {
  name: '',
  industry: '',
  stage: '',
  employees: 0,
  foundedYear: new Date().getFullYear(),
  location: '',
  ceo: '',
  investment: 0,
};

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

export function AddCompanyDialog() {
  const [formData, setFormData] = useState<CompanyFormData>(initialFormData);
  const [isOpen, setIsOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      setIsSubmitting(true);

      // 企業スコアは初期値として70〜90の間の乱数を設定
      const initialScore = Math.floor(Math.random() * 21) + 70;

      // APIを使用して企業を追加
      const companyData = {
        name: formData.name,
        industry: formData.industry,
        stage: formData.stage,
        employees: formData.employees,
        foundedYear: formData.foundedYear,
        location: formData.location,
        score: initialScore, // 初期ウェルネススコア
      };

      const newCompany = await companiesApi.addCompany(companyData);

      toast({
        title: '企業を登録しました',
        description: `${formData.name}を登録しました。`,
      });

      setIsOpen(false);
      setFormData(initialFormData);
      navigate(`/companies/${newCompany.id}`);
    } catch (error) {
      toast({
        title: 'エラー',
        description: '企業の登録に失敗しました。',
        variant: 'destructive',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button className="w-full">
          <PlusCircle className="h-4 w-4 mr-2" />
          企業を追加
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>新規企業登録</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">企業名 *</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="industry">業界 *</Label>
              <Input
                id="industry"
                value={formData.industry}
                onChange={(e) => setFormData({ ...formData, industry: e.target.value })}
                placeholder="例: SaaS, フィンテック"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="stage">ステージ *</Label>
              <Select
                value={formData.stage}
                onValueChange={(value) => setFormData({ ...formData, stage: value })}
                required
              >
                <SelectTrigger>
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
              <Label htmlFor="employees">従業員数</Label>
              <Input
                id="employees"
                type="number"
                min="1"
                value={formData.employees}
                onChange={(e) => setFormData({ ...formData, employees: parseInt(e.target.value) })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="foundedYear">設立年</Label>
              <Input
                id="foundedYear"
                type="number"
                min="1900"
                max={new Date().getFullYear()}
                value={formData.foundedYear}
                onChange={(e) => setFormData({ ...formData, foundedYear: parseInt(e.target.value) })}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="location">所在地</Label>
              <Input
                id="location"
                value={formData.location}
                onChange={(e) => setFormData({ ...formData, location: e.target.value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="investment">調達額（円）</Label>
              <Input
                id="investment"
                type="number"
                min="0"
                step="1000000"
                value={formData.investment}
                onChange={(e) => setFormData({ ...formData, investment: parseInt(e.target.value) })}
                placeholder="例: 100000000"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="ceo">代表者名</Label>
            <Input
              id="ceo"
              value={formData.ceo}
              onChange={(e) => setFormData({ ...formData, ceo: e.target.value })}
            />
          </div>

          <div className="flex justify-end space-x-2 pt-4">
            <Button variant="outline" type="button" onClick={() => setIsOpen(false)}>
              キャンセル
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  登録中...
                </>
              ) : '登録'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}