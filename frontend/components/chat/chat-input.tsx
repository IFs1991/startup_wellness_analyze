"use client";

import React, { useState, useRef, KeyboardEvent, useEffect } from 'react';
import { Send, Loader2, Paperclip, X, Mic, WifiOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';
import { FileAttachment, type FileAttachment as FileAttachmentType } from './file-attachment';
import { AnalysisRequestButton } from './analysis-request-button';
import { useOfflineQueue } from "@/lib/hooks/use-offline-queue";
import { OfflineQueueService, getOfflineQueueService } from "@/lib/offline-queue";
import { useToast } from "@/components/ui/use-toast";

export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  attachments?: FileAttachmentType[];
}

interface ChatInputProps {
  onSendMessage: (message: string, attachments?: FileAttachmentType[]) => void;
  onRequestAnalysis?: (analysisType: string, parameters?: Record<string, any>) => void;
  isDisabled?: boolean;
  isProcessing?: boolean;
  placeholder?: string;
  className?: string;
}

export function ChatInput({
  onSendMessage,
  onRequestAnalysis,
  isDisabled = false,
  isProcessing = false,
  placeholder = 'メッセージを入力...',
  className
}: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [attachments, setAttachments] = useState<FileAttachmentType[]>([]);
  const [showAttachmentPicker, setShowAttachmentPicker] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { isOnline, enqueueMessage } = useOfflineQueue({
    onOnlineStatusChange: (status) => {
      if (status) {
        toast({
          title: "ネットワーク接続が回復しました",
        });
      } else {
        toast({
          title: "オフラインモードになりました。メッセージは保存され、接続が回復したときに送信されます",
          variant: "default",
        });
      }
    }
  });

  // キャッシュされたメッセージの数を管理する状態
  const [queuedMessageCount, setQueuedMessageCount] = useState<number>(0);
  const { toast } = useToast();
  const offlineQueueService = getOfflineQueueService();

  // 定期的にオフラインキューの状態を確認
  useEffect(() => {
    const checkQueueInterval = setInterval(() => {
      setQueuedMessageCount(offlineQueueService.getQueueLength());
    }, 5000);

    // 初回実行
    setQueuedMessageCount(offlineQueueService.getQueueLength());

    return () => clearInterval(checkQueueInterval);
  }, []);

  const handleSubmit = async () => {
    const trimmedMessage = message.trim();
    if ((!trimmedMessage && attachments.length === 0) || isDisabled || isProcessing) return;

    setIsSending(true);

    try {
      if (isOnline) {
        onSendMessage(trimmedMessage, attachments.length > 0 ? attachments : undefined);
      } else {
        // オフライン時はキューに追加
        const messageId = enqueueMessage({
          type: "chat_message",
          content: trimmedMessage,
          attachments: attachments.length > 0 ? attachments.map(file => ({
            id: file.id,
            name: file.file.name,
            size: file.file.size,
            type: file.file.type,
            dataUrl: URL.createObjectURL(file.file) // ローカルでのプレビュー用URL
          })) : undefined,
          timestamp: new Date().toISOString()
        });

        // UIには即時反映（楽観的UIアップデート）
        onSendMessage(trimmedMessage, attachments.length > 0 ? attachments : undefined);

        toast({
          title: "メッセージはオフラインキャッシュに保存されました。ネットワーク接続が回復したら自動的に送信されます",
          id: `offline-message-${messageId}`,
          duration: 3000
        });
      }

      setMessage('');
      setAttachments([]);
      setShowAttachmentPicker(false);

      // フォーカスを入力欄に戻す
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    } catch (error) {
      console.error("メッセージ送信エラー:", error);
      toast({
        title: "メッセージの送信に失敗しました。もう一度お試しください",
        variant: "destructive",
      });
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Ctrl+Enterまたは⌘+Enterでメッセージを送信
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileSelect = (files: FileAttachmentType[]) => {
    setAttachments(prev => [...prev, ...files]);
  };

  const handleFileRemove = (fileId: string) => {
    setAttachments(prev => prev.filter(file => file.id !== fileId));
  };

  const toggleAttachmentPicker = () => {
    setShowAttachmentPicker(prev => !prev);
  };

  // 分析リクエストの処理
  const handleAnalysisRequest = (analysisType: string, parameters?: Record<string, any>) => {
    if (onRequestAnalysis) {
      onRequestAnalysis(analysisType, parameters);
    } else {
      // onRequestAnalysis が提供されていない場合は、代わりに通常のメッセージとして送信
      const analysisRequestMessage = `分析リクエスト: ${analysisType}\n${
        parameters ? `パラメータ: ${JSON.stringify(parameters, null, 2)}` : ''
      }`;
      onSendMessage(analysisRequestMessage);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const fileList = Array.from(e.target.files);
      setAttachments(prev => [...prev, ...fileList.map(file => ({
        id: crypto.randomUUID(),
        file: file,
        name: file.name,
        size: file.size,
        type: file.type
      }))]);
    }
  };

  return (
    <div className={cn("flex flex-col w-full", className)}>
      {/* キャッシュされたメッセージの通知 */}
      {queuedMessageCount > 0 && (
        <div className="flex items-center gap-2 mb-2 text-sm text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/30 rounded px-3 py-1">
          <WifiOff className="h-4 w-4" />
          <span>オフラインキャッシュに{queuedMessageCount}件のメッセージが保存されています</span>
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto h-6 text-xs"
            onClick={() => toast({
              title: "オフラインキャッシュの状態",
              description: `${queuedMessageCount}件のメッセージが接続回復時に自動送信されます`,
            })}
          >
            詳細
          </Button>
        </div>
      )}

      {/* 選択された添付ファイル表示 */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 p-2 bg-muted/20 rounded-t-md">
          {attachments.map((file, index) => (
            <div key={index} className="flex items-center gap-2 bg-background p-1 rounded text-xs">
              <span className="max-w-[100px] truncate">{file.file.name}</span>
              <Button
                variant="ghost"
                size="sm"
                className="h-5 w-5 p-0"
                onClick={() => handleFileRemove(file.id)}
              >
                ×
              </Button>
            </div>
          ))}
        </div>
      )}

      {/* ファイル添付ピッカー */}
      {showAttachmentPicker && (
        <div className="p-3 border border-border bg-background rounded-t-md">
          <FileAttachment
            onFileSelect={handleFileSelect}
            onFileRemove={handleFileRemove}
            attachments={[]}
            multiple={true}
            maxSize={10}
          />
        </div>
      )}

      {/* メッセージ入力エリア */}
      <div className={cn(
        "flex items-end gap-2 p-3 border border-border",
        (attachments.length > 0 || showAttachmentPicker) ? "border-t-0 rounded-b-md" : "rounded-md",
        !isOnline && "border-orange-300 dark:border-orange-600"
      )}>
        <Button
          onClick={toggleAttachmentPicker}
          variant="ghost"
          size="icon"
          className={cn(
            "h-8 w-8",
            showAttachmentPicker && "bg-muted"
          )}
          disabled={isDisabled}
        >
          <Paperclip className="h-4 w-4" />
          <span className="sr-only">ファイルを添付</span>
        </Button>

        <Textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={isOnline ? placeholder : "オフラインモード: メッセージは保存されます"}
          disabled={isDisabled}
          className={cn(
            "min-h-[80px] resize-none flex-1",
            !isOnline && "bg-orange-50 dark:bg-orange-950/10"
          )}
          rows={3}
        />

        {/* 分析リクエストボタン */}
        {onRequestAnalysis && (
          <div className="flex mb-1">
            <AnalysisRequestButton
              onRequestAnalysis={handleAnalysisRequest}
              disabled={isDisabled || isProcessing}
            />
          </div>
        )}

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          multiple
          className="hidden"
        />

        <Button
          onClick={handleSubmit}
          disabled={(message.trim() === '' && attachments.length === 0) || isDisabled || isProcessing || isSending}
          size="icon"
          className={cn(
            "h-10 w-10",
            !isOnline && "bg-orange-500 hover:bg-orange-600 text-white"
          )}
        >
          {isProcessing || isSending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : !isOnline ? (
            <WifiOff className="h-4 w-4" />
          ) : (
            <Send className="h-4 w-4" />
          )}
          <span className="sr-only">{isOnline ? "送信" : "オフラインで保存"}</span>
        </Button>
      </div>

      {!isOnline && (
        <p className="text-xs text-orange-600 dark:text-orange-400 mt-1 flex items-center">
          <WifiOff className="h-3 w-3 mr-1" />
          オフラインモード: メッセージはローカルに保存され、接続回復時に自動送信されます
        </p>
      )}
    </div>
  );
}