"use client";

import React from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ChevronDown, ChevronUp, ExternalLink, Info } from 'lucide-react';
import { Insight } from '@/lib/ai-insights-generator';
import { cn } from '@/lib/utils';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface AnalysisResultCardProps {
  title: string;
  description?: string;
  insights: Insight[];
  analysisType: string;
  className?: string;
}

export function AnalysisResultCard({
  title,
  description,
  insights,
  analysisType,
  className
}: AnalysisResultCardProps) {
  const [expanded, setExpanded] = React.useState(false);

  // インサイトを関連性スコアで並べ替え
  const sortedInsights = [...insights].sort((a, b) => b.relevanceScore - a.relevanceScore);

  // 最初の2つのインサイトのみを表示（展開時はすべて表示）
  const displayedInsights = expanded ? sortedInsights : sortedInsights.slice(0, 2);

  return (
    <Card className={cn("w-full overflow-hidden", className)}>
      <CardHeader className="bg-primary/5 p-4 pb-2">
        <CardTitle className="text-md flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span>{title}</span>
            <Badge variant="outline" className="ml-2">
              {analysisType}
            </Badge>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded(!expanded)}
            className="h-8 w-8 p-0"
          >
            {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            <span className="sr-only">{expanded ? '折りたたむ' : '展開する'}</span>
          </Button>
        </CardTitle>
        {description && <p className="text-sm text-muted-foreground mt-1">{description}</p>}
      </CardHeader>

      <CardContent className="p-0">
        <ul className="divide-y">
          {displayedInsights.map((insight) => (
            <li key={insight.id} className="p-3">
              <div className="flex gap-2 items-start">
                <div className={cn(
                  "w-2 h-2 rounded-full mt-2",
                  insight.type === 'positive' && "bg-green-500",
                  insight.type === 'negative' && "bg-red-500",
                  insight.type === 'warning' && "bg-yellow-500",
                  insight.type === 'neutral' && "bg-blue-500"
                )} />
                <div className="flex-1">
                  <h4 className="font-medium text-sm">{insight.title}</h4>
                  <p className="text-sm text-muted-foreground mt-1">{insight.description}</p>

                  {insight.glossaryTerms && Object.keys(insight.glossaryTerms).length > 0 && (
                    <div className="flex gap-1 flex-wrap mt-2">
                      {Object.entries(insight.glossaryTerms).map(([term, definition]) => (
                        <TooltipProvider key={term}>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Badge variant="secondary" className="cursor-help">
                                {term} <Info className="h-3 w-3 ml-1" />
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p className="max-w-xs">{definition}</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      ))}
                    </div>
                  )}
                </div>
                <Badge className="mt-1" variant="outline">
                  {insight.relevanceScore}%
                </Badge>
              </div>
            </li>
          ))}
        </ul>

        {!expanded && insights.length > 2 && (
          <div className="px-4 py-2 text-center">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setExpanded(true)}
              className="text-xs"
            >
              {insights.length - 2}件の追加インサイトを表示
            </Button>
          </div>
        )}
      </CardContent>

      <CardFooter className="p-3 pt-2 bg-primary/5 flex justify-between">
        <div className="text-xs text-muted-foreground">
          {insights.length}件のインサイト
        </div>
        <Button variant="outline" size="sm" className="h-7 text-xs">
          <span>詳細分析</span>
          <ExternalLink className="ml-1 h-3 w-3" />
        </Button>
      </CardFooter>
    </Card>
  );
}