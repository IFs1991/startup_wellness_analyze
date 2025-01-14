import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";

interface CompanyFiltersProps {
  industry: string;
  setIndustry: (value: string) => void;
  stage: string;
  setStage: (value: string) => void;
  scoreRange: string;
  setScoreRange: (value: string) => void;
  onReset: () => void;
}

const industries = ['すべて', 'SaaS', 'ヘルスケア', 'クリーンテック', 'フィンテック'];
const stages = ['すべて', 'シード', 'シリーズA', 'シリーズB', 'シリーズC以降'];
const scoreRanges = [
  { value: 'all', label: 'すべて' },
  { value: '90-100', label: '90-100 優秀' },
  { value: '80-89', label: '80-89 良好' },
  { value: '70-79', label: '70-79 平均的' },
  { value: '0-69', label: '69以下 要改善' },
];

export function CompanyFilters({
  industry,
  setIndustry,
  stage,
  setStage,
  scoreRange,
  setScoreRange,
  onReset,
}: CompanyFiltersProps) {
  const hasFilters = industry !== 'すべて' || stage !== 'すべて' || scoreRange !== 'all';

  return (
    <div className="flex flex-wrap gap-4 items-center">
      <Select value={industry} onValueChange={setIndustry}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="業界" />
        </SelectTrigger>
        <SelectContent>
          {industries.map((item) => (
            <SelectItem key={item} value={item}>{item}</SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Select value={stage} onValueChange={setStage}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="ステージ" />
        </SelectTrigger>
        <SelectContent>
          {stages.map((item) => (
            <SelectItem key={item} value={item}>{item}</SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Select value={scoreRange} onValueChange={setScoreRange}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="スコア範囲" />
        </SelectTrigger>
        <SelectContent>
          {scoreRanges.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {hasFilters && (
        <Button variant="ghost" size="sm" onClick={onReset}>
          <X className="h-4 w-4 mr-2" />
          フィルターをリセット
        </Button>
      )}
    </div>
  );
}