/**
 * チャット関連の型定義
 */

// メッセージの送信者タイプ
export type MessageSender = 'user' | 'ai' | 'system';

// メッセージ状態
export type MessageStatus = 'sending' | 'sent' | 'received' | 'error';

// 添付ファイルの型
export interface Attachment {
  id: string;
  name: string;
  type: string;
  url: string;
  size: number;
}

// チャットメッセージの型
export interface ChatMessage {
  id: string;
  content: string;
  sender: MessageSender;
  timestamp: Date;
  status: MessageStatus;
  attachments?: Attachment[];
  metadata?: {
    analysisType?: string;
    relatedCompanyId?: string;
    relatedReportId?: string;
    suggestion?: boolean;
    [key: string]: any;
  };
}

// チャット履歴の型
export interface ChatHistory {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
  pinned?: boolean;
}

// チャットの状態を表す型
export interface ChatState {
  activeChat: ChatHistory | null;
  chats: ChatHistory[];
  isLoading: boolean;
  error: Error | null;
}

// AIによる分析提案の型
export interface AnalysisSuggestion {
  id: string;
  type: string;
  title: string;
  description: string;
  parameters?: Record<string, any>;
}

// チャットの初期メッセージの型
export interface SuggestedQuery {
  id: string;
  text: string;
  category?: string;
}