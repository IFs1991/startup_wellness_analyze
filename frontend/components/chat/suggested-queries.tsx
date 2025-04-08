"use client";

import React from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { SuggestedQuery } from '@/types/chat';
import { cn } from '@/lib/utils';
import { Sparkles, TrendingUp, Building2, LineChart } from 'lucide-react';

interface SuggestedQueriesProps {
  queries: SuggestedQuery[];
  onSelectQuery: (query: string) => void;
  className?: string;
}

// カテゴリに応じたアイコンを表示
const getCategoryIcon = (category?: string) => {
  switch (category) {
    case 'analysis':
      return <LineChart className="h-4 w-4 mr-2" />;
    case 'trend':
      return <TrendingUp className="h-4 w-4 mr-2" />;
    case 'company':
      return <Building2 className="h-4 w-4 mr-2" />;
    default:
      return <Sparkles className="h-4 w-4 mr-2" />;
  }
};

export function SuggestedQueries({
  queries,
  onSelectQuery,
  className
}: SuggestedQueriesProps) {
  if (!queries.length) return null;

  return (
    <Card className={cn("p-4", className)}>
      <h3 className="text-sm font-medium mb-3">質問例</h3>
      <div className="flex flex-wrap gap-2">
        {queries.map((query) => (
          <Button
            key={query.id}
            variant="outline"
            size="sm"
            className="flex items-center text-xs justify-start h-auto py-2 px-3"
            onClick={() => onSelectQuery(query.text)}
          >
            {getCategoryIcon(query.category)}
            <span className="text-left">{query.text}</span>
          </Button>
        ))}
      </div>
    </Card>
  );
}

// よく使われるサンプルクエリのリスト
export const SAMPLE_QUERIES: SuggestedQuery[] = [
  {
    id: '1',
    text: '当社の組織ウェルネススコアはどうなっていますか？',
    category: 'analysis'
  },
  {
    id: '2',
    text: '業界平均と比較した当社の従業員満足度は？',
    category: 'trend'
  },
  {
    id: '3',
    text: '離職率を低下させるための具体的な施策を教えてください',
    category: 'analysis'
  },
  {
    id: '4',
    text: '組織文化の健全性を評価する主要な指標は何ですか？',
    category: 'analysis'
  },
  {
    id: '5',
    text: '競合他社と比較した当社の人事施策の強みと弱みは？',
    category: 'company'
  },
  {
    id: '6',
    text: 'リモートワークが従業員エンゲージメントに与える影響を分析してください',
    category: 'trend'
  },
];