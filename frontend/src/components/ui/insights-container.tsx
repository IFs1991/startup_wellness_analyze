import React, { useState, useEffect } from 'react';
import { InsightCard, InsightCardProps } from './insight-card';
import { Button } from './button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from './card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './select';
import {
  Filter,
  SortDesc,
  SortAsc,
  Lightbulb,
  LoaderCircle,
  Loader2,
  Grid3X3,
  List,
  CheckCircle,
  AlertCircle,
  AlertTriangle,
  Info,
  Sparkles
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './tabs';
import { Switch } from './switch';
import { Label } from './label';
import { Separator } from './separator';
import { ScrollArea } from './scroll-area';

export interface Insight {
  id: string;
  title: string;
  description: string;
  type: 'positive' | 'negative' | 'warning' | 'neutral';
  relevanceScore: number;
  aiGenerated: boolean;
  relatedMetric?: string;
  glossaryTerms?: Record<string, string>;
}

export interface InsightsContainerProps {
  insights: Insight[];
  title?: string;
  description?: string;
  isLoading?: boolean;
  onGenerateAiInsights?: () => Promise<void>;
  onInsightFeedback?: (insightId: string, isHelpful: boolean) => void;
  onInsightClick?: (insightId: string) => void;
  className?: string;
  isGeneratingAiInsights?: boolean;
}

export const InsightsContainer: React.FC<InsightsContainerProps> = ({
  insights,
  title = "分析インサイト",
  description = "データから抽出された重要な情報と知見",
  isLoading = false,
  onGenerateAiInsights,
  onInsightFeedback,
  onInsightClick,
  className,
  isGeneratingAiInsights = false
}) => {
  // 表示オプションの状態管理
  const [viewType, setViewType] = useState<'grid' | 'list'>('grid');
  const [expandedInsightId, setExpandedInsightId] = useState<string | null>(null);

  // フィルタリングオプションの状態管理
  const [filters, setFilters] = useState({
    positive: true,
    negative: true,
    neutral: true,
    warning: true,
    aiGenerated: true,
    humanCreated: true
  });

  // ソートオプションの状態管理
  const [sortBy, setSortBy] = useState<'relevance' | 'type'>('relevance');

  // フィルタリングされたインサイトのリスト
  const filteredAndSortedInsights = React.useMemo(() => {
    // フィルタリング
    let result = insights.filter(insight => {
      if (insight.type === 'positive' && !filters.positive) return false;
      if (insight.type === 'negative' && !filters.negative) return false;
      if (insight.type === 'neutral' && !filters.neutral) return false;
      if (insight.type === 'warning' && !filters.warning) return false;

      if (insight.aiGenerated && !filters.aiGenerated) return false;
      if (!insight.aiGenerated && !filters.humanCreated) return false;

      return true;
    });

    // ソート
    result = [...result].sort((a, b) => {
      if (sortBy === 'relevance') {
        return b.relevanceScore - a.relevanceScore;
      } else {
        // typeでソート（positive, warning, neutral, negative の順）
        const typeOrder = { positive: 0, warning: 1, neutral: 2, negative: 3 };
        return typeOrder[a.type] - typeOrder[b.type];
      }
    });

    return result;
  }, [insights, filters, sortBy]);

  // インサイトカードがクリックされたときの処理
  const handleInsightClick = (insightId: string) => {
    if (expandedInsightId === insightId) {
      setExpandedInsightId(null);
    } else {
      setExpandedInsightId(insightId);
    }

    if (onInsightClick) {
      onInsightClick(insightId);
    }
  };

  // フィルタの変更を管理する関数
  const toggleFilter = (filterName: keyof typeof filters) => {
    setFilters(prev => ({
      ...prev,
      [filterName]: !prev[filterName]
    }));
  };

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl">{title}</CardTitle>

          <div className="flex items-center gap-2">
            <Tabs
              defaultValue="view"
              className="h-8"
            >
              <TabsList className="h-8">
                <TabsTrigger
                  value="view"
                  className="h-8 px-2.5"
                  onClick={() => setViewType('grid')}
                >
                  <Grid3X3 className="h-4 w-4" />
                </TabsTrigger>
                <TabsTrigger
                  value="list"
                  className="h-8 px-2.5"
                  onClick={() => setViewType('list')}
                >
                  <List className="h-4 w-4" />
                </TabsTrigger>
              </TabsList>
            </Tabs>

            <Select
              defaultValue="relevance"
              onValueChange={(value) => setSortBy(value as 'relevance' | 'type')}
            >
              <SelectTrigger className="h-8 w-[130px]">
                <div className="flex items-center gap-2">
                  <SortDesc className="h-3.5 w-3.5" />
                  <SelectValue placeholder="並び替え" />
                </div>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="relevance">関連性</SelectItem>
                <SelectItem value="type">種類</SelectItem>
              </SelectContent>
            </Select>

            <Button
              variant="outline"
              size="sm"
              className="h-8"
              onClick={() => {
                const sheet = document.getElementById('filter-sheet');
                if (sheet) {
                  sheet.classList.toggle('hidden');
                }
              }}
            >
              <Filter className="h-3.5 w-3.5 mr-1.5" />
              フィルター
            </Button>
          </div>
        </div>

        <CardDescription>{description}</CardDescription>

        {/* フィルタリングシート */}
        <div id="filter-sheet" className="hidden mt-3 border rounded-md p-3 bg-background">
          <div className="flex justify-between items-center mb-2">
            <h4 className="text-sm font-medium">表示するインサイトをフィルタ</h4>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2"
              onClick={() => {
                const sheet = document.getElementById('filter-sheet');
                if (sheet) {
                  sheet.classList.add('hidden');
                }
              }}
            >
              閉じる
            </Button>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <h5 className="text-xs font-medium mb-2">インサイトタイプ</h5>
              <div className="space-y-2">
                <div className="flex items-center">
                  <Switch
                    checked={filters.positive}
                    onCheckedChange={() => toggleFilter('positive')}
                    id="filter-positive"
                  />
                  <Label htmlFor="filter-positive" className="ml-2 flex items-center">
                    <CheckCircle className="h-3.5 w-3.5 text-green-500 mr-1.5" />
                    <span>ポジティブ</span>
                  </Label>
                </div>
                <div className="flex items-center">
                  <Switch
                    checked={filters.negative}
                    onCheckedChange={() => toggleFilter('negative')}
                    id="filter-negative"
                  />
                  <Label htmlFor="filter-negative" className="ml-2 flex items-center">
                    <AlertCircle className="h-3.5 w-3.5 text-red-500 mr-1.5" />
                    <span>ネガティブ</span>
                  </Label>
                </div>
                <div className="flex items-center">
                  <Switch
                    checked={filters.warning}
                    onCheckedChange={() => toggleFilter('warning')}
                    id="filter-warning"
                  />
                  <Label htmlFor="filter-warning" className="ml-2 flex items-center">
                    <AlertTriangle className="h-3.5 w-3.5 text-amber-500 mr-1.5" />
                    <span>警告</span>
                  </Label>
                </div>
                <div className="flex items-center">
                  <Switch
                    checked={filters.neutral}
                    onCheckedChange={() => toggleFilter('neutral')}
                    id="filter-neutral"
                  />
                  <Label htmlFor="filter-neutral" className="ml-2 flex items-center">
                    <Info className="h-3.5 w-3.5 text-blue-500 mr-1.5" />
                    <span>中立</span>
                  </Label>
                </div>
              </div>
            </div>

            <div>
              <h5 className="text-xs font-medium mb-2">インサイトソース</h5>
              <div className="space-y-2">
                <div className="flex items-center">
                  <Switch
                    checked={filters.aiGenerated}
                    onCheckedChange={() => toggleFilter('aiGenerated')}
                    id="filter-ai"
                  />
                  <Label htmlFor="filter-ai" className="ml-2 flex items-center">
                    <Sparkles className="h-3.5 w-3.5 text-purple-500 mr-1.5" />
                    <span>AI生成</span>
                  </Label>
                </div>
                <div className="flex items-center">
                  <Switch
                    checked={filters.humanCreated}
                    onCheckedChange={() => toggleFilter('humanCreated')}
                    id="filter-human"
                  />
                  <Label htmlFor="filter-human" className="ml-2 flex items-center">
                    <span>人間作成</span>
                  </Label>
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        {onGenerateAiInsights && (
          <div className="mb-4 flex items-center">
            <Button
              variant="secondary"
              size="sm"
              className="mr-2"
              onClick={onGenerateAiInsights}
              disabled={isGeneratingAiInsights}
            >
              {isGeneratingAiInsights ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  AIインサイト生成中...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 mr-2" />
                  AIインサイトを生成
                </>
              )}
            </Button>
            <p className="text-xs text-muted-foreground">
              データを分析し、AIによる追加インサイトを生成します
            </p>
          </div>
        )}

        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
            <p className="text-sm text-muted-foreground">インサイトを読み込み中...</p>
          </div>
        ) : filteredAndSortedInsights.length === 0 ? (
          <div className="text-center py-8 border rounded-md bg-muted/20">
            <p className="text-sm text-muted-foreground">
              条件に一致するインサイトが見つかりませんでした。
              フィルターを調整してみてください。
            </p>
          </div>
        ) : (
          <ScrollArea className="max-h-[600px] pr-4 -mr-4">
            <div className={cn(
              "grid gap-3",
              viewType === 'grid' ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'
            )}>
              {filteredAndSortedInsights.map(insight => (
                <InsightCard
                  key={insight.id}
                  id={insight.id}
                  title={insight.title}
                  description={insight.description}
                  type={insight.type}
                  relevanceScore={insight.relevanceScore}
                  aiGenerated={insight.aiGenerated}
                  relatedMetric={insight.relatedMetric}
                  glossaryTerms={insight.glossaryTerms}
                  onFeedback={onInsightFeedback}
                  onClick={handleInsightClick}
                  expanded={expandedInsightId === insight.id}
                />
              ))}
            </div>
          </ScrollArea>
        )}

        {!isLoading && filteredAndSortedInsights.length > 0 && (
          <div className="mt-4 flex justify-between text-xs text-muted-foreground">
            <span>表示中: {filteredAndSortedInsights.length} / {insights.length} インサイト</span>
            {onInsightFeedback && (
              <span>フィードバックを提供して、インサイトの品質向上にご協力ください</span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};