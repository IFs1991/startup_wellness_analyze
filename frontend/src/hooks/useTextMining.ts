import { useState, useEffect, useCallback } from 'react';
import { useWebSocketConnection } from './useWebSocketConnection';
import { useToast } from './use-toast';

export interface SentimentAnalysis {
  positive: number;
  neutral: number;
  negative: number;
}

export interface KeywordAnalysis {
  word: string;
  count: number;
  sentiment: 'positive' | 'neutral' | 'negative';
}

export interface PhraseAnalysis {
  phrase: string;
  count: number;
  sentiment: 'positive' | 'neutral' | 'negative';
}

export interface EntityAnalysis {
  entity: string;
  type: string;
  count: number;
  mentions: string[];
}

export interface TopicAnalysis {
  topic: string;
  keywords: string[];
  documentCount: number;
  score: number;
}

export interface TextInsight {
  insight: string;
  confidence: number;
  relatedTexts: string[];
  source: string;
}

export interface TextAnalysisFilter {
  startDate?: string;
  endDate?: string;
  source?: string[];
  sentiment?: string[];
  keywords?: string[];
}

export interface TextMiningResult {
  sentiment?: SentimentAnalysis;
  keywords?: KeywordAnalysis[];
  phrases?: PhraseAnalysis[];
  entities?: EntityAnalysis[];
  topics?: TopicAnalysis[];
  insights?: TextInsight[];
  loading: boolean;
  error: Error | null;
}

/**
 * テキストマイニングデータを取得するカスタムフック
 *
 * @param companyId 企業ID（指定しない場合は全体データ）
 * @param textSource テキストのソース（'feedback', 'surveys', 'reviews'など）
 * @param filters 分析フィルター
 * @returns テキストマイニング結果と状態
 */
export const useTextMining = (
  companyId?: string,
  textSource: string = 'all',
  filters: TextAnalysisFilter = {}
): TextMiningResult & {
  refreshData: () => void;
  applyFilters: (filters: TextAnalysisFilter) => void;
  analyzeCustomText: (text: string) => Promise<void>;
} => {
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [currentFilters, setCurrentFilters] = useState<TextAnalysisFilter>(filters);

  const [sentiment, setSentiment] = useState<SentimentAnalysis | undefined>(undefined);
  const [keywords, setKeywords] = useState<KeywordAnalysis[] | undefined>(undefined);
  const [phrases, setPhrases] = useState<PhraseAnalysis[] | undefined>(undefined);
  const [entities, setEntities] = useState<EntityAnalysis[] | undefined>(undefined);
  const [topics, setTopics] = useState<TopicAnalysis[] | undefined>(undefined);
  const [insights, setInsights] = useState<TextInsight[] | undefined>(undefined);

  // エンドポイント作成
  const endpoint = companyId
    ? `text_mining/company/${companyId}/${textSource}`
    : `text_mining/${textSource}`;

  // WebSocket接続を確立
  const {
    status,
    sendMessage,
    messages,
    error: wsError
  } = useWebSocketConnection(endpoint);

  // エラー時の処理
  useEffect(() => {
    if (wsError) {
      setError(wsError);
      setLoading(false);

      toast({
        title: 'データ取得エラー',
        description: 'テキスト分析データの取得に失敗しました',
        variant: 'destructive'
      });
    }
  }, [wsError, toast]);

  // 接続状態の監視
  useEffect(() => {
    if (status === 'connected') {
      // 接続成功時にデータをリクエスト
      requestTextMiningData();
    } else if (status === 'disconnected' || status === 'error') {
      setLoading(false);
    }
  }, [status]);

  // テキストマイニングデータをリクエストする関数
  const requestTextMiningData = useCallback(() => {
    if (status === 'connected') {
      setLoading(true);

      const requestPayload = {
        company_id: companyId,
        text_source: textSource,
        filters: currentFilters
      };

      sendMessage('get_text_mining_data', requestPayload);
    }
  }, [status, sendMessage, companyId, textSource, currentFilters]);

  // フィルターを適用する関数
  const applyFilters = useCallback((newFilters: TextAnalysisFilter) => {
    setCurrentFilters(prev => ({
      ...prev,
      ...newFilters
    }));

    if (status === 'connected') {
      setLoading(true);

      const requestPayload = {
        company_id: companyId,
        text_source: textSource,
        filters: {
          ...currentFilters,
          ...newFilters
        }
      };

      sendMessage('get_text_mining_data', requestPayload);
    } else {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
    }
  }, [status, sendMessage, companyId, textSource, currentFilters, toast]);

  // カスタムテキストを分析する関数
  const analyzeCustomText = useCallback(async (text: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (status === 'connected') {
        setLoading(true);

        const requestPayload = {
          text,
          company_id: companyId
        };

        // 一意のメッセージIDを生成
        const messageId = Date.now().toString();

        // レスポンスを待つためのリスナー
        const messageListener = (event: MessageEvent) => {
          try {
            const data = JSON.parse(event.data);

            if (data.request_id === messageId && data.type === 'text_analysis_result') {
              // 分析結果を設定
              if (data.data.sentiment) {
                setSentiment(data.data.sentiment);
              }

              if (data.data.keywords) {
                setKeywords(data.data.keywords);
              }

              if (data.data.phrases) {
                setPhrases(data.data.phrases);
              }

              if (data.data.entities) {
                setEntities(data.data.entities);
              }

              if (data.data.topics) {
                setTopics(data.data.topics);
              }

              if (data.data.insights) {
                setInsights(data.data.insights);
              }

              setLoading(false);
              resolve();
            }
          } catch (err) {
            console.error('メッセージの解析に失敗:', err);
          }
        };

        // WebSocketのonmessageイベントにリスナーを追加
        const ws = (window as any).webSocketConnections?.[endpoint];
        if (ws) {
          ws.addEventListener('message', messageListener);

          // リクエスト送信
          sendMessage('analyze_custom_text', {
            ...requestPayload,
            request_id: messageId
          });

          // 30秒後にタイムアウト
          setTimeout(() => {
            ws.removeEventListener('message', messageListener);
            setLoading(false);

            setError(new Error('テキスト分析がタイムアウトしました'));
            toast({
              title: '分析タイムアウト',
              description: 'テキスト分析の処理に時間がかかりすぎています',
              variant: 'destructive'
            });

            reject(new Error('テキスト分析がタイムアウトしました'));
          }, 30000);
        } else {
          setLoading(false);
          reject(new Error('WebSocket接続が見つかりません'));
        }
      } else {
        toast({
          title: '接続エラー',
          description: 'サーバーに接続されていません',
          variant: 'destructive'
        });

        reject(new Error('サーバーに接続されていません'));
      }
    });
  }, [status, sendMessage, companyId, endpoint, toast]);

  // メッセージの処理
  useEffect(() => {
    if (messages && messages.length > 0) {
      const latestMessage = messages[messages.length - 1];

      if (latestMessage.type === 'text_mining_data') {
        const data = latestMessage.data;

        // 各データの設定
        if (data.sentiment) {
          setSentiment(data.sentiment);
        }

        if (data.keywords) {
          setKeywords(data.keywords);
        }

        if (data.phrases) {
          setPhrases(data.phrases);
        }

        if (data.entities) {
          setEntities(data.entities);
        }

        if (data.topics) {
          setTopics(data.topics);
        }

        if (data.insights) {
          setInsights(data.insights);
        }

        setLoading(false);
      }
    }
  }, [messages]);

  // データの更新をリクエストする関数
  const refreshData = useCallback(() => {
    if (status === 'connected') {
      requestTextMiningData();
    } else {
      toast({
        title: '接続エラー',
        description: 'サーバーに接続されていません',
        variant: 'destructive'
      });
    }
  }, [status, requestTextMiningData, toast]);

  return {
    sentiment,
    keywords,
    phrases,
    entities,
    topics,
    insights,
    loading,
    error,
    refreshData,
    applyFilters,
    analyzeCustomText
  };
};