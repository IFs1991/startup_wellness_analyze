"use client";

import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from './useAuth';
import { User } from 'firebase/auth';
import { useToast } from './use-toast';

// 環境変数のバリデーションと設定
const validateWebSocketUrl = () => {
  const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080';

  // 開発環境での警告（プロダクション環境では表示しない）
  if (process.env.NODE_ENV !== 'production') {
    if (!process.env.NEXT_PUBLIC_WS_URL) {
      console.warn('警告: NEXT_PUBLIC_WS_URL環境変数が設定されていません。デフォルト値を使用します:', WS_URL);
    }
  }

  return WS_URL;
};

// WebSocketの設定とリトライオプションのデフォルト値
const DEFAULT_WS_OPTIONS = {
  autoReconnect: true,
  reconnectInterval: 5000,
  maxReconnectAttempts: 5,
  reconnectBackoffMultiplier: 1.5,  // 指数バックオフのための乗数
  pingInterval: 30000,  // 30秒ごとにpingを送信
  pingTimeoutMs: 10000  // ping応答待機時間
};

interface WebSocketOptions {
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  reconnectBackoffMultiplier?: number;
  pingInterval?: number;
  pingTimeoutMs?: number;
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
    maxReconnectAttempts = 5,
    reconnectBackoffMultiplier = 1.5,
    pingInterval = 30000,
    pingTimeoutMs = 5000
  } = options;

  const { user } = useAuth();
  const { toast } = useToast();

  const [status, setStatus] = useState<WebSocketStatus>('disconnected');
  const [messages, setMessages] = useState<any[]>([]);
  const [error, setError] = useState<Error | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const pingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Pingを送信する関数
  const sendPing = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }));

      // Ping応答タイムアウトの設定
      if (pingTimeoutRef.current) {
        clearTimeout(pingTimeoutRef.current);
      }

      pingTimeoutRef.current = setTimeout(() => {
        console.warn('WebSocket ping timeout');
        if (wsRef.current) {
          wsRef.current.close();
        }
      }, pingTimeoutMs);
    }
  }, [pingTimeoutMs]);

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

    // Pingインターバルとタイムアウトのクリーンアップ
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    if (pingTimeoutRef.current) {
      clearTimeout(pingTimeoutRef.current);
      pingTimeoutRef.current = null;
    }

    try {
      setStatus('connecting');
      // 環境変数のバリデーション
      const WS_BASE_URL = validateWebSocketUrl();
      // wsから始まるAPIエンドポイント
      const wsUrl = `${WS_BASE_URL}/ws/${endpoint}`;
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

        // 定期的なpingの開始
        if (pingInterval > 0) {
          pingIntervalRef.current = setInterval(sendPing, pingInterval);
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

          // Pongメッセージの処理
          if (data.type === 'pong') {
            if (pingTimeoutRef.current) {
              clearTimeout(pingTimeoutRef.current);
              pingTimeoutRef.current = null;
            }
            return;
          }

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

        // Pingインターバルとタイムアウトのクリーンアップ
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        if (pingTimeoutRef.current) {
          clearTimeout(pingTimeoutRef.current);
          pingTimeoutRef.current = null;
        }

        if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          setStatus('reconnecting');
          reconnectAttemptsRef.current += 1;

          // 指数バックオフを適用したリトライ間隔
          const backoffTime = reconnectInterval * Math.pow(reconnectBackoffMultiplier, reconnectAttemptsRef.current - 1);
          console.log(`再接続を試みます (${reconnectAttemptsRef.current}/${maxReconnectAttempts}) ${backoffTime}ms後`);

          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, backoffTime);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          console.error(`最大再接続試行回数(${maxReconnectAttempts})に達しました`);
          setError(new Error('接続の再確立に失敗しました'));
          toast({
            title: '接続エラー',
            description: '接続の再確立に失敗しました。後でもう一度お試しください。',
            variant: 'destructive'
          });
        }
      };

      wsRef.current = ws;
    } catch (err) {
      setError(err instanceof Error ? err : new Error('WebSocket接続の初期化に失敗しました'));
      setStatus('error');
    }
  }, [endpoint, user, toast, autoReconnect, reconnectInterval, maxReconnectAttempts, reconnectBackoffMultiplier, pingInterval, pingTimeoutMs, sendPing]);

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

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    if (pingTimeoutRef.current) {
      clearTimeout(pingTimeoutRef.current);
      pingTimeoutRef.current = null;
    }

    setStatus('disconnected');
  }, []);

  // マウント時に接続、アンマウント時に切断
  useEffect(() => {
    // Next.jsではブラウザ環境のチェックが必要
    if (typeof window !== 'undefined') {
      connectWebSocket();
    }

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