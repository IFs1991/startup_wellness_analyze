"use client";

import React from 'react';
import { cn } from '@/lib/utils';
import { Avatar } from '../ui/avatar';
import { User, Bot, AlertCircle, RefreshCw, Paperclip, FileText, Image as ImageIcon, File } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { AnalysisResultCard } from './analysis-result-card';
import { Insight } from '@/lib/ai-insights-generator';
import { FileAttachment as FileAttachmentType } from './file-attachment';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date | string;
  status?: 'sending' | 'sent' | 'error';
  analysis?: {
    title: string;
    description?: string;
    insights: Insight[];
    analysisType: string;
  };
  attachments?: FileAttachmentType[];
}

interface ChatMessageItemProps {
  message: ChatMessage;
  className?: string;
  onRetry?: () => void;
}

export function ChatMessageItem({ message, className, onRetry }: ChatMessageItemProps) {
  const isUser = message.role === 'user';
  const isError = message.status === 'error';
  const isSending = message.status === 'sending';
  const hasAnalysis = message.analysis && message.analysis.insights.length > 0;
  const hasAttachments = message.attachments && message.attachments.length > 0;

  // タイムスタンプを文字列として処理
  const timestamp = typeof message.timestamp === 'string'
    ? new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // ファイルタイプに基づいたアイコンを返す関数
  const getFileIcon = (mimeType: string) => {
    if (mimeType.startsWith('image/')) {
      return <ImageIcon className="h-4 w-4" />;
    } else if (mimeType.startsWith('text/')) {
      return <FileText className="h-4 w-4" />;
    } else {
      return <File className="h-4 w-4" />;
    }
  };

  // マークダウンのカスタムレンダリングコンポーネント
  const MarkdownComponents = {
    // 見出しのスタイル調整
    h1: ({ node, ...props }: any) => <h1 className="text-xl font-bold my-3" {...props} />,
    h2: ({ node, ...props }: any) => <h2 className="text-lg font-bold my-2" {...props} />,
    h3: ({ node, ...props }: any) => <h3 className="text-md font-bold my-2" {...props} />,
    // リンクのスタイル調整
    a: ({ node, ...props }: any) => (
      <a
        className={cn(
          "underline hover:text-primary transition-colors",
          isUser ? "text-primary-foreground" : "text-primary"
        )}
        target="_blank"
        rel="noopener noreferrer"
        {...props}
      />
    ),
    // リストのスタイル調整
    ul: ({ node, ...props }: any) => <ul className="list-disc pl-6 my-2" {...props} />,
    ol: ({ node, ...props }: any) => <ol className="list-decimal pl-6 my-2" {...props} />,
    // コードブロックのスタイル調整
    code: ({ node, inline, ...props }: any) => (
      inline
        ? <code className={cn("px-1 py-0.5 rounded bg-muted text-sm font-mono", isUser ? "bg-primary-foreground/20" : "bg-background")} {...props} />
        : <code className={cn("block p-2 my-2 rounded bg-muted/70 text-sm font-mono whitespace-pre-wrap overflow-x-auto", isUser ? "bg-primary-foreground/20" : "bg-background")} {...props} />
    ),
    // 表のスタイル調整
    table: ({ node, ...props }: any) => (
      <div className="overflow-x-auto my-2">
        <table className="min-w-full border-collapse" {...props} />
      </div>
    ),
    tr: ({ node, ...props }: any) => <tr className="border-b border-muted" {...props} />,
    th: ({ node, ...props }: any) => <th className="px-2 py-1 text-left font-medium" {...props} />,
    td: ({ node, ...props }: any) => <td className="px-2 py-1" {...props} />,
  };

  return (
    <div
      className={cn(
        "flex gap-3 w-full group",
        isUser ? "justify-end" : "justify-start",
        className
      )}
    >
      {!isUser && (
        <Avatar className="h-8 w-8 bg-primary/10">
          <Bot className="h-4 w-4 text-primary" />
        </Avatar>
      )}

      <div className={cn(
        "rounded-lg px-4 py-2 max-w-[80%] relative",
        isUser
          ? "bg-primary text-primary-foreground"
          : "bg-muted",
        isError && "border border-destructive",
        hasAnalysis && "w-full max-w-[90%]"
      )}>
        {/* 添付ファイルがある場合、先に表示 */}
        {hasAttachments && (
          <div className={cn(
            "mb-2 flex flex-wrap gap-2",
            message.content ? "pb-2 border-b" : ""
          )}>
            {message.attachments?.map((attachment) => (
              <div
                key={attachment.id}
                className={cn(
                  "flex items-center gap-1 p-2 rounded-md text-sm",
                  isUser ? "bg-primary-foreground/10" : "bg-background/60"
                )}
              >
                {attachment.previewUrl && attachment.file.type.startsWith('image/') ? (
                  <a
                    href={attachment.previewUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block"
                  >
                    <img
                      src={attachment.previewUrl}
                      alt={attachment.file.name}
                      className="max-h-32 max-w-32 rounded-md object-cover"
                    />
                  </a>
                ) : (
                  <div className={cn(
                    "flex items-center gap-1 text-xs",
                    isUser ? "text-primary-foreground" : "text-foreground"
                  )}>
                    {getFileIcon(attachment.file.type)}
                    <span className="max-w-[120px] truncate">{attachment.file.name}</span>
                    <span className="text-xs opacity-70">
                      {(attachment.file.size / 1024).toFixed(0)}KB
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {message.content && (
          <div className="break-words markdown-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeRaw]}
              components={MarkdownComponents}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* 分析結果カードの表示 */}
        {hasAnalysis && message.analysis && (
          <div className="mt-3 pt-3 border-t">
            <AnalysisResultCard
              title={message.analysis.title}
              description={message.analysis.description}
              insights={message.analysis.insights}
              analysisType={message.analysis.analysisType}
              className="bg-background/60 backdrop-blur-sm"
            />
          </div>
        )}

        <div className={cn(
          "text-xs mt-1 flex items-center gap-1",
          isUser ? "text-primary-foreground/80" : "text-muted-foreground"
        )}>
          {timestamp}
          {hasAttachments && (
            <span className="inline-flex items-center gap-1">
              <Paperclip className="h-3 w-3" />
              {message.attachments?.length}
            </span>
          )}
          {isSending && <span className="inline-flex items-center">· 送信中...</span>}
          {isError && (
            <span className="inline-flex items-center text-destructive gap-1">
              <AlertCircle className="h-3 w-3" /> 送信エラー
            </span>
          )}
        </div>

        {isError && onRetry && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onRetry}
            className="absolute top-1 right-1 h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <RefreshCw className="h-3 w-3" />
            <span className="sr-only">再送信</span>
          </Button>
        )}
      </div>

      {isUser && (
        <Avatar className="h-8 w-8 bg-primary">
          <User className="h-4 w-4 text-primary-foreground" />
        </Avatar>
      )}
    </div>
  );
}