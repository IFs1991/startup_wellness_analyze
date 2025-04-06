import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from './useAuth';
import { User } from 'firebase/auth'; // firebase/auth から直接インポート
import { useToast } from './use-toast';

interface WebSocketOptions {
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'reconnecting' | 'error';

/**
 * WebSocket接続を管理するカスタムフック
 *
 * @param endpoint WebSocketのエンドポイントパス（例: 'dashboard', 'company/123'）
 * @param options 接続オプション
 * @returns WebSocket関連の状態と操作関数
 */
export const useWebSocketConnection = (
  endpoint: string,
  options: WebSocketOptions = {}
) => {
  const {
    autoReconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 5
  } = options;

  const { user } = useAuth();
  const { toast } = useToast();

  const [status, setStatus] = useState<WebSocketStatus>('disconnected');
  const [messages, setMessages] = useState<any[]>([]);
  const [error, setError] = useState<Error | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocketを初期化する関数
  const connectWebSocket = useCallback(async () => {
    if (!user) {
      setError(new Error('認証情報がありません'));
      return;
    }

    // 既存の接続をクリーンアップ
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      setStatus('connecting');
      // wsから始まるAPIエンドポイント
      const wsUrl = `ws://localhost:8080/ws/${endpoint}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = async () => {
        setStatus('connected');
        reconnectAttemptsRef.current = 0;

        // 認証情報を送信
        try {
          const token = await (user as any).getIdToken();
          ws.send(JSON.stringify({
            type: 'init',
            token: token
          }));
        } catch (tokenError) {
          console.error('トークンの取得に失敗:', tokenError);
          setError(new Error('認証トークンの取得に失敗しました'));
          ws.close();
          setStatus('error');
          toast({
            title: '認証エラー',
            description: '認証情報の取得に失敗しました',
            variant: 'destructive'
          });
          return;
        }

        toast({
          title: '接続成功',
          description: 'リアルタイムデータに接続しました',
          variant: 'default'
        });
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setMessages(prev => [...prev, data]);
        } catch (err) {
          console.error('メッセージの解析に失敗:', err);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocketエラー:', event);
        setStatus('error');
        setError(new Error('WebSocket接続でエラーが発生しました'));

        toast({
          title: '接続エラー',
          description: 'サーバーとの接続に問題が発生しました',
          variant: 'destructive'
        });
      };

      ws.onclose = () => {
        setStatus('disconnected');

        if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          setStatus('reconnecting');
          reconnectAttemptsRef.current += 1;

          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, reconnectInterval);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      setError(err instanceof Error ? err : new Error('WebSocket接続の初期化に失敗しました'));
      setStatus('error');
    }
  }, [endpoint, user, toast, autoReconnect, reconnectInterval, maxReconnectAttempts]);

  // メッセージを送信する関数
  const sendMessage = useCallback((type: string, payload?: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const message = {
        type,
        ...payload
      };
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  // 特定タイプのメッセージだけを取得
  const getMessagesByType = useCallback((type: string) => {
    return messages.filter(msg => msg.type === type);
  }, [messages]);

  // 最新のメッセージを取得
  const getLatestMessage = useCallback((type?: string) => {
    if (type) {
      const filtered = messages.filter(msg => msg.type === type);
      return filtered.length > 0 ? filtered[filtered.length - 1] : null;
    }
    return messages.length > 0 ? messages[messages.length - 1] : null;
  }, [messages]);

  // WebSocketを切断する関数
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    setStatus('disconnected');
  }, []);

  // マウント時に接続、アンマウント時に切断
  useEffect(() => {
    connectWebSocket();

    return () => {
      disconnect();
    };
  }, [connectWebSocket, disconnect]);

  return {
    status,
    messages,
    error,
    sendMessage,
    getMessagesByType,
    getLatestMessage,
    reconnect: connectWebSocket,
    disconnect
  };
};