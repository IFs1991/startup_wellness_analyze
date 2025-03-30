import React, { useState } from 'react';
import { Card, CardContent, CardFooter, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { ChevronDown, ChevronUp, ThumbsUp, ThumbsDown, AlertTriangle, CheckCircle,
  Info, AlertCircle, Lightbulb, Sparkles, HelpCircle, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface InsightCardProps {
  id: string;
  title: string;
  description: string;
  type: 'positive' | 'negative' | 'warning' | 'neutral';
  relevanceScore: number;  // 0-100の範囲
  aiGenerated?: boolean;   // AIによって生成されたかどうか
  relatedMetric?: string;  // 関連する指標
  glossaryTerms?: Record<string, string>; // 用語集
  onFeedback?: (insightId: string, isHelpful: boolean) => void;
  onClick?: (insightId: string) => void;
  expanded?: boolean;
  className?: string;
}

export const InsightCard: React.FC<InsightCardProps> = ({
  id,
  title,
  description,
  type,
  relevanceScore,
  aiGenerated = false,
  relatedMetric,
  glossaryTerms = {},
  onFeedback,
  onClick,
  expanded: initialExpanded = false,
  className
}) => {
  const [expanded, setExpanded] = useState(initialExpanded);
  const [feedbackGiven, setFeedbackGiven] = useState<'up' | 'down' | null>(null);

  // 形式ごとのアイコンとカラー
  const typeConfig = {
    positive: {
      icon: <CheckCircle className="h-5 w-5 text-green-500" />,
      color: 'bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800',
      header: 'text-green-700 dark:text-green-300',
      badge: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
    },
    negative: {
      icon: <AlertCircle className="h-5 w-5 text-red-500" />,
      color: 'bg-red-50 border-red-200 dark:bg-red-950 dark:border-red-800',
      header: 'text-red-700 dark:text-red-300',
      badge: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
    },
    warning: {
      icon: <AlertTriangle className="h-5 w-5 text-amber-500" />,
      color: 'bg-amber-50 border-amber-200 dark:bg-amber-950 dark:border-amber-800',
      header: 'text-amber-700 dark:text-amber-300',
      badge: 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300'
    },
    neutral: {
      icon: <Info className="h-5 w-5 text-blue-500" />,
      color: 'bg-blue-50 border-blue-200 dark:bg-blue-950 dark:border-blue-800',
      header: 'text-blue-700 dark:text-blue-300',
      badge: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300'
    }
  };

  // 関連性スコアの視覚的表現を計算
  const getRelevanceIndicator = () => {
    if (relevanceScore >= 90) return '非常に高い';
    if (relevanceScore >= 80) return '高い';
    if (relevanceScore >= 70) return '中程度';
    if (relevanceScore >= 60) return '普通';
    return '低い';
  };

  // 説明文のハイライト：用語集の単語をハイライトして、ツールチップを追加
  const renderDescriptionWithTooltips = () => {
    if (!glossaryTerms || Object.keys(glossaryTerms).length === 0) {
      return description;
    }

    let result = description;
    const elements: React.ReactNode[] = [];
    let lastIndex = 0;

    // 用語集の単語を正規表現でマッチするための準備
    const terms = Object.keys(glossaryTerms).sort((a, b) => b.length - a.length); // 長い単語から先にマッチ
    const termsRegex = new RegExp(`(${terms.map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'g');

    // マッチした単語をハイライトしてツールチップを追加
    const matches = [...result.matchAll(termsRegex)];

    if (matches.length === 0) {
      elements.push(result);
    } else {
      matches.forEach((match, i) => {
        const term = match[0];
        const index = match.index || 0;

        // マッチした単語の前のテキストを追加
        if (index > lastIndex) {
          elements.push(result.substring(lastIndex, index));
        }

        // マッチした単語をツールチップ付きで追加
        elements.push(
          <TooltipProvider key={`term-${i}`}>
            <Tooltip delayDuration={300}>
              <TooltipTrigger asChild>
                <span className="underline decoration-dotted underline-offset-2 cursor-help font-medium">
                  {term}
                </span>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-xs">
                <p className="text-sm">{glossaryTerms[term]}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );

        lastIndex = index + term.length;
      });

      // 最後のマッチの後のテキストを追加
      if (lastIndex < result.length) {
        elements.push(result.substring(lastIndex));
      }
    }

    return <>{elements}</>;
  };

  const handleFeedback = (isHelpful: boolean) => {
    if (onFeedback) {
      onFeedback(id, isHelpful);
      setFeedbackGiven(isHelpful ? 'up' : 'down');
    }
  };

  return (
    <Card
      className={cn(
        "overflow-hidden border transition-all duration-300",
        typeConfig[type].color,
        expanded ? "shadow-md" : "shadow-sm",
        onClick ? "cursor-pointer hover:shadow-md" : "",
        className
      )}
      onClick={onClick ? () => onClick(id) : undefined}
    >
      <CardHeader className={cn("py-3 px-4 flex flex-row items-center gap-2 bg-opacity-40", typeConfig[type].header)}>
        {typeConfig[type].icon}
        <div className="flex-1 font-medium text-sm sm:text-base">{title}</div>
        {aiGenerated && (
          <Badge variant="outline" className="flex items-center gap-1 ml-2 font-normal">
            <Sparkles className="h-3 w-3" />
            <span className="text-xs">AI</span>
          </Badge>
        )}
        {relatedMetric && (
          <Badge variant="secondary" className={cn("text-xs", typeConfig[type].badge)}>
            {relatedMetric}
          </Badge>
        )}
      </CardHeader>

      <CardContent className={cn(
        "px-4 transition-all duration-300 overflow-hidden text-sm",
        expanded ? "py-4" : "py-3"
      )}>
        <div className={cn(
          "transition-all duration-300",
          !expanded && description.length > 150 ? "line-clamp-2" : ""
        )}>
          {renderDescriptionWithTooltips()}
        </div>

        {Object.keys(glossaryTerms).length > 0 && expanded && (
          <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
            <div className="text-xs font-medium flex items-center gap-1 mb-2 text-gray-500 dark:text-gray-400">
              <HelpCircle className="h-3.5 w-3.5" />
              用語集
            </div>
            <div className="grid grid-cols-1 gap-2 text-xs">
              {Object.entries(glossaryTerms).map(([term, definition], index) => (
                <div key={index} className="flex flex-col">
                  <span className="font-medium">{term}</span>
                  <span className="text-gray-600 dark:text-gray-400">{definition}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>

      <CardFooter className="py-2 px-4 bg-gray-50 dark:bg-gray-900 flex justify-between items-center">
        <div className="flex items-center text-xs text-gray-500 dark:text-gray-400">
          <Lightbulb className="h-3.5 w-3.5 mr-1" />
          <span>関連性:</span>
          <Badge variant="outline" className="ml-1.5 font-normal">
            {getRelevanceIndicator()} ({relevanceScore}%)
          </Badge>
        </div>

        <div className="flex items-center gap-2">
          {description.length > 150 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-8 px-2 text-gray-500"
              onClick={(e) => {
                e.stopPropagation();
                setExpanded(!expanded);
              }}
            >
              {expanded ? (
                <div className="flex items-center">
                  <span className="text-xs mr-1">折りたたむ</span>
                  <ChevronUp className="h-4 w-4" />
                </div>
              ) : (
                <div className="flex items-center">
                  <span className="text-xs mr-1">もっと見る</span>
                  <ChevronDown className="h-4 w-4" />
                </div>
              )}
            </Button>
          )}

          {onFeedback && (
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                className={cn(
                  "h-7 w-7",
                  feedbackGiven === 'up' ? "text-green-600 bg-green-50 dark:bg-green-900" : "text-gray-500"
                )}
                disabled={feedbackGiven !== null}
                onClick={(e) => {
                  e.stopPropagation();
                  handleFeedback(true);
                }}
              >
                <ThumbsUp className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className={cn(
                  "h-7 w-7",
                  feedbackGiven === 'down' ? "text-red-600 bg-red-50 dark:bg-red-900" : "text-gray-500"
                )}
                disabled={feedbackGiven !== null}
                onClick={(e) => {
                  e.stopPropagation();
                  handleFeedback(false);
                }}
              >
                <ThumbsDown className="h-4 w-4" />
              </Button>
            </div>
          )}

          {onClick && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-gray-500"
              onClick={(e) => {
                e.stopPropagation();
                onClick(id);
              }}
            >
              <ExternalLink className="h-4 w-4" />
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  );
};