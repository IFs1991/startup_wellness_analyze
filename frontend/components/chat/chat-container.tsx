"use client";

import React, { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';
import { ChatMessage, ChatMessageItem } from './chat-message';
import { Skeleton } from '@/components/ui/skeleton';

interface ChatContainerProps {
  messages: ChatMessage[];
  className?: string;
  onRetry?: (messageId: string) => void;
  loading?: boolean;
}

export function ChatContainer({
  messages,
  className,
  onRetry,
  loading = false
}: ChatContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // 新しいメッセージが追加されたら、自動的に最新のメッセージにスクロール
  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [messages]);

  // メッセージがない場合のプレースホルダー
  if (messages.length === 0 && !loading) {
    return (
      <div className={cn(
        "flex flex-col items-center justify-center h-full p-8 text-center text-muted-foreground",
        className
      )}>
        <p>会話を開始するためにメッセージを送信してください。</p>
        <p className="text-sm mt-2">健康やウェルネスに関する質問から始めましょう。</p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        "flex flex-col gap-4 p-4 overflow-y-auto",
        className
      )}
    >
      {/* メッセージリスト */}
      {messages.map((message) => (
        <ChatMessageItem
          key={message.id}
          message={message}
          onRetry={message.status === 'error' ? () => onRetry?.(message.id) : undefined}
        />
      ))}

      {/* ローディング中のスケルトン表示 */}
      {loading && (
        <div className="flex gap-3">
          <Skeleton className="h-8 w-8 rounded-full" />
          <div className="space-y-2">
            <Skeleton className="h-4 w-[250px]" />
            <Skeleton className="h-4 w-[200px]" />
          </div>
        </div>
      )}
    </div>
  );
}