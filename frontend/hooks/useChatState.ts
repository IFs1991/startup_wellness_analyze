"use client";

import { useState, useCallback, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useAuth } from './useAuth';
import { useToast } from './useToast';
import { useWebSocketConnection } from './useWebSocketConnection';
import { FileAttachment } from '@/components/chat/file-attachment';
import { OfflineQueueService } from '@/lib/offline-queue';

// ローカルストレージのキー
const CHAT_STORAGE_KEY = 'wellness-analyzer-chat-history';

// チャットメッセージの型定義
export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  status?: 'sending' | 'sent' | 'error';
  analysis?: {
    title: string;
    description?: string;
    insights: any[];
    analysisType: string;
  };
  attachments?: FileAttachment[];
}

// チャット状態フックの戻り値の型
export interface UseChatStateReturn {
  messages: ChatMessage[];
  isProcessing: boolean;
  error: Error | null;
  addUserMessage: (content: string, attachments?: FileAttachment[]) => string; // 返り値はメッセージID
  addAssistantMessage: (
    content: string,
    analysis?: {
      title: string;
      description?: string;
      insights: any[];
      analysisType: string;
    },
    attachments?: FileAttachment[]
  ) => void;
  updateMessageStatus: (id: string, status: 'sending' | 'sent' | 'error') => void;
  clearMessages: () => void;
  setIsProcessing: (isProcessing: boolean) => void;
  retryMessage: (id: string) => void;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting';
  reconnect: () => void;
}

/**
 * チャットの状態を管理するカスタムフック
 * WebSocket接続とメッセージのやり取りを処理する
 */
export function useChatState(): UseChatStateReturn {
  const { user } = useAuth();
  const { toast } = useToast();
  const [messages, setMessages] = useState<ChatMessage[]>([{
    id: 'welcome',
    role: 'assistant',
    content: 'こんにちは、スタートアップウェルネスアナライザーへようこそ。健康や組織のウェルネスについて質問してください。',
    timestamp: new Date(),
  }]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const initialLoadRef = useRef(false);
  const offlineQueueRef = useRef<OfflineQueueService>(OfflineQueueService.getInstance());

  // WebSocket接続
  const {
    status: connectionStatus,
    sendMessage,
    reconnect: reconnectWebSocket,
    error: wsError
  } = useWebSocketConnection('chat', {
    autoReconnect: true,
    reconnectInterval: 3000,
    maxReconnectAttempts: 10
  });

  // ローカルストレージからチャット履歴を読み込む
  const loadMessages = useCallback(() => {
    if (typeof window === 'undefined' || !user) return;

    try {
      const storedMessages = localStorage.getItem(`${CHAT_STORAGE_KEY}-${user.uid}`);
      if (storedMessages) {
        const parsedMessages: ChatMessage[] = JSON.parse(storedMessages);

        // 日付文字列をDateオブジェクトに変換
        const formattedMessages = parsedMessages.map(message => ({
          ...message,
          timestamp: new Date(message.timestamp)
        }));

        setMessages(formattedMessages);
      }
    } catch (err) {
      console.error('チャット履歴の読み込みに失敗:', err);
      setError(err instanceof Error ? err : new Error('チャット履歴の読み込みに失敗しました'));

      toast({
        title: 'エラー',
        description: 'チャット履歴の読み込みに失敗しました',
        variant: 'destructive'
      });
    }
  }, [user, toast]);

  // ローカルストレージにチャット履歴を保存
  const saveMessages = useCallback((messagesToSave: ChatMessage[]) => {
    if (typeof window === 'undefined' || !user) return;

    try {
      // 添付ファイルオブジェクトはシリアライズできないので、保存前に変換
      const messagesToStore = messagesToSave
        .filter(m => m.id !== 'welcome')
        .slice(-100)
        .map(message => {
          if (!message.attachments) return message;

          // 添付ファイルの必要な情報だけを保存
          const simplifiedAttachments = message.attachments.map(att => ({
            id: att.id,
            name: att.file.name,
            type: att.file.type,
            size: att.file.size,
            // previewUrlはローカルストレージには保存しない
          }));

          return {
            ...message,
            attachments: simplifiedAttachments
          };
        });

      localStorage.setItem(`${CHAT_STORAGE_KEY}-${user.uid}`, JSON.stringify(messagesToStore));
    } catch (err) {
      console.error('チャット履歴の保存に失敗:', err);
    }
  }, [user]);

  // ユーザーが変わったときにチャット履歴を読み込む
  useEffect(() => {
    if (user && !initialLoadRef.current) {
      loadMessages();
      initialLoadRef.current = true;
    }
  }, [user, loadMessages]);

  // WebSocketエラーを監視
  useEffect(() => {
    if (wsError) {
      setError(wsError);
      toast({
        title: '接続エラー',
        description: 'チャットサーバーへの接続に問題が発生しました。再接続を試みています。',
        variant: 'destructive'
      });
    }
  }, [wsError, toast]);

  // オフラインキューの初期化と状態監視
  useEffect(() => {
    if (!user) return;

    const offlineQueue = offlineQueueRef.current;

    // オフラインキューから未送信メッセージを復元
    const restoreQueuedMessages = () => {
      const queuedMessages = offlineQueue.getQueue();
      if (queuedMessages.length > 0) {
        toast({
          title: '未送信メッセージ',
          description: `${queuedMessages.length}件の未送信メッセージがあります。オンラインになると自動的に送信されます。`,
          variant: 'default'
        });
      }
    };

    // ネットワーク状態の変化を監視
    const handleNetworkChange = (isOnline: boolean) => {
      if (isOnline) {
        // オンラインになったらキューにあるメッセージを処理
        processQueuedMessages();
      } else {
        toast({
          title: 'オフライン',
          description: 'ネットワーク接続がありません。メッセージはオンラインになったときに送信されます。',
          variant: 'default'
        });
      }
    };

    // キューにあるメッセージを処理
    const processQueuedMessages = async () => {
      const queue = offlineQueue.getQueue();
      if (queue.length === 0) return;

      toast({
        title: 'メッセージ同期中',
        description: `${queue.length}件のメッセージを送信しています...`,
        variant: 'default'
      });

      for (const item of queue) {
        try {
          // メッセージのステータスを「送信中」に更新
          updateMessageStatus(item.messageId, 'sending');

          // WebSocketでメッセージを送信
          if (connectionStatus === 'connected') {
            sendMessage('chat_message', {
              message: item.content,
              messageId: item.messageId,
              attachments: item.attachments
            });

            // 送信成功としてマーク
            updateMessageStatus(item.messageId, 'sent');

            // キューから削除
            offlineQueue.removeFromQueue(item.messageId);
          } else {
            // まだ接続が復帰していない場合は中断
            break;
          }
        } catch (err) {
          console.error('メッセージの再送信に失敗:', err);
          updateMessageStatus(item.messageId, 'error');
        }
      }

      // キューの残りを確認
      const remainingQueue = offlineQueue.getQueue();
      if (remainingQueue.length > 0) {
        toast({
          title: '一部のメッセージが未送信',
          description: `${remainingQueue.length}件のメッセージが未送信です。接続が安定したら再試行します。`,
          variant: 'destructive'
        });
      } else {
        toast({
          title: '同期完了',
          description: 'すべてのメッセージが同期されました',
          variant: 'default'
        });
      }
    };

    // 初期ロード時にキューを確認
    restoreQueuedMessages();

    // オンライン/オフラインイベントのリスナーを設定
    window.addEventListener('online', () => handleNetworkChange(true));
    window.addEventListener('offline', () => handleNetworkChange(false));

    // WebSocket接続状態の変化を監視
    if (connectionStatus === 'connected') {
      processQueuedMessages();
    }

    return () => {
      window.removeEventListener('online', () => handleNetworkChange(true));
      window.removeEventListener('offline', () => handleNetworkChange(false));
    };
  }, [connectionStatus, sendMessage, toast, user]);

  // ユーザーメッセージを追加
  const addUserMessage = useCallback((content: string, attachments: FileAttachment[] = []): string => {
    const messageId = uuidv4();
    const newMessage: ChatMessage = {
      id: messageId,
      content,
      role: 'user',
      timestamp: new Date(),
      status: navigator.onLine ? 'sending' : 'error',
      attachments
    };

    setMessages(prev => {
      const updatedMessages = [...prev, newMessage];
      saveMessages(updatedMessages);
      return updatedMessages;
    });

    // オンラインの場合はWebSocketで送信、オフラインの場合はキューに追加
    if (navigator.onLine && connectionStatus === 'connected') {
      sendMessage('chat_message', {
        message: content,
        messageId,
        attachments: attachments.map(att => ({
          id: att.id,
          name: att.file.name,
          type: att.file.type,
          size: att.file.size
        }))
      });
      updateMessageStatus(messageId, 'sent');
    } else {
      // オフラインまたは接続が確立していない場合はキューに追加
      offlineQueueRef.current.addToQueue({
        messageId,
        content,
        timestamp: Date.now(),
        retryCount: 0,
        attachments: attachments.map(att => ({
          id: att.id,
          name: att.file.name,
          type: att.file.type,
          size: att.file.size
        }))
      });

      toast({
        title: 'オフラインモード',
        description: 'メッセージはオンラインになったときに送信されます',
        variant: 'default'
      });
    }

    return messageId;
  }, [connectionStatus, saveMessages, sendMessage, toast]);

  // アシスタントメッセージを追加
  const addAssistantMessage = useCallback((
    content: string,
    analysis?: {
      title: string;
      description?: string;
      insights: any[];
      analysisType: string;
    },
    attachments: FileAttachment[] = []
  ): void => {
    const newMessage: ChatMessage = {
      id: uuidv4(),
      content,
      role: 'assistant',
      timestamp: new Date(),
      analysis,
      attachments
    };

    setMessages(prev => {
      const updatedMessages = [...prev, newMessage];
      saveMessages(updatedMessages);
      return updatedMessages;
    });
  }, [saveMessages]);

  // メッセージステータスを更新
  const updateMessageStatus = useCallback((id: string, status: 'sending' | 'sent' | 'error'): void => {
    setMessages(prev => {
      const updatedMessages = prev.map(msg =>
        msg.id === id ? { ...msg, status } : msg
      );
      saveMessages(updatedMessages);
      return updatedMessages;
    });
  }, [saveMessages]);

  // メッセージを再送信
  const retryMessage = useCallback((id: string): void => {
    const message = messages.find(msg => msg.id === id);
    if (!message) return;

    // メッセージのステータスを「送信中」に更新
    updateMessageStatus(id, 'sending');

    if (navigator.onLine && connectionStatus === 'connected') {
      // WebSocketでメッセージを再送信
      sendMessage('chat_message', {
        message: message.content,
        messageId: id,
        attachments: message.attachments?.map(att => ({
          id: att.id,
          name: att.file.name,
          type: att.file.type,
          size: att.file.size
        }))
      });
      updateMessageStatus(id, 'sent');

      // キューから削除（もし存在していれば）
      offlineQueueRef.current.removeFromQueue(id);
    } else {
      // まだオフラインの場合はキューに追加/更新
      offlineQueueRef.current.addToQueue({
        messageId: id,
        content: message.content,
        timestamp: message.timestamp.getTime(),
        retryCount: 0,
        attachments: message.attachments?.map(att => ({
          id: att.id,
          name: att.file.name,
          type: att.file.type,
          size: att.file.size
        }))
      });

      toast({
        title: 'オフラインモード',
        description: 'メッセージはオンラインになったときに送信されます',
        variant: 'default'
      });
    }
  }, [connectionStatus, messages, sendMessage, toast, updateMessageStatus]);

  // メッセージをクリア
  const clearMessages = useCallback((): void => {
    // ウェルカムメッセージのみを残す
    setMessages([{
      id: 'welcome',
      role: 'assistant',
      content: 'こんにちは、スタートアップウェルネスアナライザーへようこそ。健康や組織のウェルネスについて質問してください。',
      timestamp: new Date(),
    }]);

    // ローカルストレージからも削除
    if (typeof window !== 'undefined' && user) {
      localStorage.removeItem(`${CHAT_STORAGE_KEY}-${user.uid}`);
    }

    // オフラインキューもクリア
    offlineQueueRef.current.clearQueue();
  }, [user]);

  // WebSocket接続を再確立
  const reconnect = useCallback((): void => {
    reconnectWebSocket();
  }, [reconnectWebSocket]);

  return {
    messages,
    isProcessing,
    error,
    addUserMessage,
    addAssistantMessage,
    updateMessageStatus,
    clearMessages,
    setIsProcessing,
    retryMessage,
    connectionStatus,
    reconnect
  };
}