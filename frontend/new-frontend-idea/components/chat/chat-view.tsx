"use client";

import React, { useRef, useEffect } from 'react';
import { ChatMessageItem } from './chat-message';
import { ChatInput } from './chat-input';
import { SuggestedQueries, SAMPLE_QUERIES } from './suggested-queries';
import { useChat } from '@/hooks/useChat';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { PlusCircle, List, History } from 'lucide-react';

export function ChatView() {
  const {
    activeChat,
    chats,
    createChat,
    selectChat,
    deleteChat,
    sendMessage,
    isLoading,
    wsStatus
  } = useChat();

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 新しいメッセージが追加されたら自動スクロール
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [activeChat?.messages]);

  const handleSendMessage = (content: string) => {
    sendMessage(content);
  };

  const handleSelectQuery = (query: string) => {
    handleSendMessage(query);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex justify-between items-center border-b p-4">
        <div className="flex items-center">
          <h2 className="text-xl font-semibold mr-2">
            {activeChat?.title || '新しい会話'}
          </h2>
          {wsStatus === 'connecting' && (
            <div className="text-xs text-muted-foreground flex items-center">
              <Loader2 className="h-3 w-3 animate-spin mr-1" />
              接続中...
            </div>
          )}
          {wsStatus === 'connected' && (
            <div className="h-2 w-2 rounded-full bg-green-500" title="接続済み" />
          )}
          {(wsStatus === 'error' || wsStatus === 'disconnected') && (
            <div className="text-xs text-destructive flex items-center">
              <span className="h-2 w-2 rounded-full bg-destructive mr-1" />
              接続エラー
            </div>
          )}
        </div>
        <div className="flex space-x-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={createChat}
            title="新しい会話"
          >
            <PlusCircle className="h-5 w-5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            title="会話履歴"
          >
            <History className="h-5 w-5" />
          </Button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* サイドバー（モバイルでは非表示） */}
        <div className="hidden md:block w-64 border-r overflow-y-auto">
          <div className="p-3">
            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={createChat}
            >
              <PlusCircle className="h-4 w-4 mr-2" />
              新しい会話
            </Button>
          </div>
          <div className="space-y-1 px-2">
            {chats.map((chat) => (
              <Button
                key={chat.id}
                variant={activeChat?.id === chat.id ? "secondary" : "ghost"}
                className="w-full justify-start text-sm h-auto py-2 px-3 overflow-hidden"
                onClick={() => selectChat(chat.id)}
              >
                <div className="flex-1 text-left truncate">
                  {chat.title}
                </div>
              </Button>
            ))}
          </div>
        </div>

        {/* メインチャットエリア */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* メッセージエリア */}
          <ScrollArea className="flex-1 p-4">
            {activeChat && activeChat.messages.length > 0 ? (
              <div className="space-y-4">
                {activeChat.messages.map((message) => (
                  <ChatMessageItem
                    key={message.id}
                    message={message}
                  />
                ))}
                <div ref={messagesEndRef} />
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center">
                <div className="max-w-md w-full space-y-4 p-4">
                  <div className="text-center mb-6">
                    <h2 className="text-2xl font-bold mb-2">スタートアップウェルネスアナライザー</h2>
                    <p className="text-muted-foreground">
                      組織健全性の分析と改善提案をAIがサポートします。
                      どのようなことでもお気軽にお尋ねください。
                    </p>
                  </div>
                  <SuggestedQueries
                    queries={SAMPLE_QUERIES}
                    onSelectQuery={handleSelectQuery}
                  />
                </div>
              </div>
            )}
          </ScrollArea>

          {/* 入力エリア */}
          <div className="p-4 border-t">
            <ChatInput
              onSendMessage={handleSendMessage}
              isLoading={isLoading || wsStatus !== 'connected'}
              disabled={wsStatus !== 'connected'}
              placeholder={
                wsStatus !== 'connected'
                  ? '接続中... しばらくお待ちください'
                  : 'メッセージを入力してください...'
              }
            />
          </div>
        </div>
      </div>
    </div>
  );
}