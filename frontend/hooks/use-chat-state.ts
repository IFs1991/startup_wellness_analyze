"use client";

import { useState, useCallback, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { useAuth } from './useAuth';
import { useToast } from './use-toast';
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

  // メッセージのステータスを更新
  const updateMessageStatus = useCallback((id: string, status: 'sending' | 'sent' | 'error') => {
    setMessages(prev => {
      const updatedMessages = prev.map(message =>
        message.id === id
          ? { ...message, status }
          : message
      );
      saveMessages(updatedMessages);
      return updatedMessages;
    });
  }, [saveMessages]);

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
        title: 'メッセージ送信中',
        description: `${queue.length}件のメッセージを送信しています...`,
        variant: 'default'
      });

      for (const queuedMessage of queue) {
        // 再試行回数が多すぎる場合はスキップ
        if (queuedMessage.retryCount > 3) {
          updateMessageStatus(queuedMessage.messageId, 'error');
          offlineQueue.removeFromQueue(queuedMessage.messageId);
          continue;
        }

        // UIのステータスを更新
        updateMessageStatus(queuedMessage.messageId, 'sending');

        // メッセージ送信
        try {
          // QueuedMessageインターフェースに合わせて修正
          sendMessage('chat_message', {
            content: queuedMessage.content,
            attachments: queuedMessage.attachments
          });
          // 送信成功としてマーク
          updateMessageStatus(queuedMessage.messageId, 'sent');
          // キューから削除
          offlineQueue.removeFromQueue(queuedMessage.messageId);
        } catch (err) {
          // 送信失敗、リトライカウントを増やす
          // プライベートメソッドの代わりに、キューから削除して再追加する方法を使用
          offlineQueue.removeFromQueue(queuedMessage.messageId);
          offlineQueue.addToQueue({
            ...queuedMessage,
            retryCount: queuedMessage.retryCount + 1
          });
          updateMessageStatus(queuedMessage.messageId, 'error');
        }
      }
    };

    // イベントリスナーを設定
    window.addEventListener('online', () => handleNetworkChange(true));
    window.addEventListener('offline', () => handleNetworkChange(false));

    // 初期化時に未送信メッセージを復元
    restoreQueuedMessages();

    return () => {
      // クリーンアップ: ネットワーク状態監視を解除
      window.removeEventListener('online', () => handleNetworkChange(true));
      window.removeEventListener('offline', () => handleNetworkChange(false));
    };
  }, [user, toast, updateMessageStatus, sendMessage]);

  // ユーザーメッセージを追加（オフライン対応版）
  const addUserMessage = useCallback((content: string, attachments?: FileAttachment[]): string => {
    const messageId = uuidv4();

    const newMessage: ChatMessage = {
      id: messageId,
      content,
      role: 'user',
      timestamp: new Date(),
      status: 'sending',
      attachments
    };

    setMessages(prev => {
      const newMessages = [...prev, newMessage];
      saveMessages(newMessages);
      return newMessages;
    });

    const messagePayload = {
      content,
      attachments: attachments ? attachments.map(att => ({
        id: att.id,
        name: att.file.name,
        type: att.file.type,
        size: att.file.size
      })) : undefined
    };

    // オフライン状態またはWebSocket接続が切断されている場合
    if (!navigator.onLine || connectionStatus !== 'connected') {
      // キューに追加
      offlineQueueRef.current.addToQueue({
        messageId,
        content,
        timestamp: Date.now(), // Date型からnumber型に修正
        retryCount: 0,
        attachments: attachments ? attachments.map(att => ({
          id: att.id,
          name: att.file.name,
          type: att.file.type,
          size: att.file.size
        })) : undefined
      });

      // UIに通知
      toast({
        title: 'オフラインモード',
        description: 'メッセージはオンラインになったときに送信されます',
        variant: 'default'
      });

      // 送信ステータスをエラーに更新
      updateMessageStatus(messageId, 'error');
    } else {
      // オンラインの場合は通常送信
      try {
        sendMessage('chat_message', messagePayload);
        // 送信成功としてマーク（実際には非同期だが簡略化）
        setTimeout(() => {
          updateMessageStatus(messageId, 'sent');
        }, 500);
      } catch (error) {
        // 送信エラー、キューに追加
        offlineQueueRef.current.addToQueue({
          messageId,
          content,
          timestamp: Date.now(), // Date型からnumber型に修正
          retryCount: 0,
          attachments: attachments ? attachments.map(att => ({
            id: att.id,
            name: att.file.name,
            type: att.file.type,
            size: att.file.size
          })) : undefined
        });
        updateMessageStatus(messageId, 'error');
      }
    }

    return messageId;
  }, [saveMessages, connectionStatus, sendMessage, updateMessageStatus, toast]);

  // アシスタントメッセージを追加
  const addAssistantMessage = useCallback((
    content: string,
    analysis?: {
      title: string;
      description?: string;
      insights: any[];
      analysisType: string;
    },
    attachments?: FileAttachment[]
  ) => {
    const newMessage: ChatMessage = {
      id: uuidv4(),
      content,
      role: 'assistant',
      timestamp: new Date(),
      analysis,
      attachments
    };

    setMessages(prev => {
      const newMessages = [...prev, newMessage];
      saveMessages(newMessages);
      return newMessages;
    });
  }, [saveMessages]);

  // メッセージを再送信（オフライン対応版）
  const retryMessage = useCallback((id: string) => {
    // 対象のメッセージを検索
    const messageToRetry = messages.find(m => m.id === id);
    if (!messageToRetry || messageToRetry.role !== 'user') return;

    // ステータスを送信中に更新
    updateMessageStatus(id, 'sending');
    setError(null);
    setIsProcessing(true);

    const messagePayload = {
      content: messageToRetry.content,
      attachments: messageToRetry.attachments ? messageToRetry.attachments.map(att => ({
        id: att.id,
        name: att.file.name,
        type: att.file.type,
        size: att.file.size
      })) : undefined
    };

    // オフライン状態またはWebSocket接続が切断されている場合
    if (!navigator.onLine || connectionStatus !== 'connected') {
      // キューに追加
      offlineQueueRef.current.addToQueue({
        messageId: id,
        content: messageToRetry.content,
        timestamp: Date.now(), // Date型からnumber型に修正
        retryCount: 0,
        attachments: messageToRetry.attachments ? messageToRetry.attachments.map(att => ({
          id: att.id,
          name: att.file.name,
          type: att.file.type,
          size: att.file.size
        })) : undefined
      });

      // UIに通知
      toast({
        title: 'オフラインモード',
        description: 'メッセージはオンラインになったときに送信されます',
        variant: 'default'
      });

      // ステータスをエラーに更新
      updateMessageStatus(id, 'error');
    } else {
      // オンラインの場合は通常送信
      try {
        sendMessage('chat_message', messagePayload);
        // 送信成功としてマーク
        setTimeout(() => {
          updateMessageStatus(id, 'sent');
        }, 500);
      } catch (error) {
        // 送信エラー、キューに追加
        offlineQueueRef.current.addToQueue({
          messageId: id,
          content: messageToRetry.content,
          timestamp: Date.now(), // Date型からnumber型に修正
          retryCount: 0,
          attachments: messageToRetry.attachments ? messageToRetry.attachments.map(att => ({
            id: att.id,
            name: att.file.name,
            type: att.file.type,
            size: att.file.size
          })) : undefined
        });
        updateMessageStatus(id, 'error');
      }
    }
  }, [messages, updateMessageStatus, sendMessage, connectionStatus, toast]);

  // すべてのメッセージをクリア
  const clearMessages = useCallback(() => {
    setMessages([{
      id: 'welcome',
      role: 'assistant',
      content: 'こんにちは、スタートアップウェルネスアナライザーへようこそ。健康や組織のウェルネスについて質問してください。',
      timestamp: new Date(),
    }]);
    setError(null);

    // ローカルストレージからも削除
    if (typeof window !== 'undefined' && user) {
      localStorage.removeItem(`${CHAT_STORAGE_KEY}-${user.uid}`);
    }
  }, [user]);

  // WebSocket再接続
  const reconnect = useCallback(() => {
    setError(null);
    reconnectWebSocket();
    toast({
      title: '再接続中',
      description: 'チャットサーバーに再接続しています...',
      variant: 'default'
    });
  }, [reconnectWebSocket, toast]);

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