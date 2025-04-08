"use client";

import { useState, useCallback, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  ChatMessage,
  ChatHistory,
  ChatState,
  MessageSender,
  MessageStatus,
  Attachment
} from '@/types/chat';
import { useAuth } from './useAuth';
import { useToast } from './use-toast';
import { useWebSocketConnection } from './useWebSocketConnection';
import { User } from 'firebase/auth';

// ローカルストレージのキー
const CHAT_STORAGE_KEY = 'startup_wellness_chats';

/**
 * チャット機能を管理するカスタムフック
 */
export const useChat = () => {
  const { user } = useAuth();
  const { toast } = useToast();
  const [state, setState] = useState<ChatState>({
    activeChat: null,
    chats: [],
    isLoading: true,
    error: null
  });

  // WebSocket接続
  const ws = useWebSocketConnection('chat');

  // チャット履歴をローカルストレージから読み込む
  const loadChats = useCallback(() => {
    if (typeof window === 'undefined' || !user) return;

    try {
      setState(prev => ({ ...prev, isLoading: true }));
      const storedChats = localStorage.getItem(CHAT_STORAGE_KEY);

      if (storedChats) {
        const parsedChats: ChatHistory[] = JSON.parse(storedChats);

        // 日付文字列をDateオブジェクトに変換
        const formattedChats = parsedChats.map(chat => ({
          ...chat,
          createdAt: new Date(chat.createdAt),
          updatedAt: new Date(chat.updatedAt),
          messages: chat.messages.map(message => ({
            ...message,
            timestamp: new Date(message.timestamp)
          }))
        }));

        setState(prev => ({
          ...prev,
          chats: formattedChats,
          activeChat: formattedChats.length > 0 ? formattedChats[0] : null,
          isLoading: false
        }));
      } else {
        setState(prev => ({ ...prev, isLoading: false }));
      }
    } catch (error) {
      console.error('チャット履歴の読み込みに失敗:', error);
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error : new Error('チャット履歴の読み込みに失敗しました'),
        isLoading: false
      }));

      toast({
        title: 'エラー',
        description: 'チャット履歴の読み込みに失敗しました',
        variant: 'destructive'
      });
    }
  }, [user, toast]);

  // チャット履歴をローカルストレージに保存
  const saveChats = useCallback((chats: ChatHistory[]) => {
    if (typeof window === 'undefined') return;

    try {
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chats));
    } catch (error) {
      console.error('チャット履歴の保存に失敗:', error);
      toast({
        title: 'エラー',
        description: 'チャット履歴の保存に失敗しました',
        variant: 'destructive'
      });
    }
  }, [toast]);

  // 新しいチャットを作成
  const createChat = useCallback(() => {
    const newChat: ChatHistory = {
      id: uuidv4(),
      title: `新しい会話 ${new Date().toLocaleString('ja-JP', { month: 'numeric', day: 'numeric', hour: 'numeric', minute: 'numeric' })}`,
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };

    setState(prev => {
      const updatedChats = [newChat, ...prev.chats];
      saveChats(updatedChats);
      return {
        ...prev,
        chats: updatedChats,
        activeChat: newChat
      };
    });

    return newChat;
  }, [saveChats]);

  // チャットを選択
  const selectChat = useCallback((chatId: string) => {
    setState(prev => {
      const selectedChat = prev.chats.find(chat => chat.id === chatId) || null;
      return {
        ...prev,
        activeChat: selectedChat
      };
    });
  }, []);

  // チャットを削除
  const deleteChat = useCallback((chatId: string) => {
    setState(prev => {
      const updatedChats = prev.chats.filter(chat => chat.id !== chatId);
      saveChats(updatedChats);

      return {
        ...prev,
        chats: updatedChats,
        activeChat: prev.activeChat?.id === chatId
          ? (updatedChats.length > 0 ? updatedChats[0] : null)
          : prev.activeChat
      };
    });
  }, [saveChats]);

  // チャットの名前を変更
  const renameChat = useCallback((chatId: string, newTitle: string) => {
    setState(prev => {
      const updatedChats = prev.chats.map(chat => {
        if (chat.id === chatId) {
          return {
            ...chat,
            title: newTitle,
            updatedAt: new Date()
          };
        }
        return chat;
      });

      saveChats(updatedChats);

      return {
        ...prev,
        chats: updatedChats,
        activeChat: prev.activeChat?.id === chatId
          ? { ...prev.activeChat, title: newTitle, updatedAt: new Date() }
          : prev.activeChat
      };
    });
  }, [saveChats]);

  // メッセージを送信
  const sendMessage = useCallback(async (content: string, attachments: Attachment[] = []) => {
    if (!state.activeChat) {
      const newChat = createChat();

      const newMessage: ChatMessage = {
        id: uuidv4(),
        content,
        sender: 'user',
        timestamp: new Date(),
        status: 'sending',
        attachments
      };

      setState(prev => {
        const updatedChat = {
          ...newChat,
          messages: [newMessage],
          updatedAt: new Date()
        };

        const updatedChats = prev.chats.map(chat =>
          chat.id === updatedChat.id ? updatedChat : chat
        );

        saveChats(updatedChats);

        return {
          ...prev,
          chats: updatedChats,
          activeChat: updatedChat
        };
      });

      // WebSocketを使用してメッセージを送信
      const messageSent = ws.sendMessage('chat_message', {
        chatId: newChat.id,
        message: {
          id: newMessage.id,
          content: newMessage.content,
          attachments: newMessage.attachments
        }
      });

      if (messageSent) {
        // 送信成功時のステータス更新
        setState(prev => {
          if (!prev.activeChat) return prev;

          const updatedMessages = prev.activeChat.messages.map(msg =>
            msg.id === newMessage.id ? { ...msg, status: 'sent' as MessageStatus } : msg
          );

          const updatedChat = {
            ...prev.activeChat,
            messages: updatedMessages
          };

          const updatedChats = prev.chats.map(chat =>
            chat.id === updatedChat.id ? updatedChat : chat
          );

          saveChats(updatedChats);

          return {
            ...prev,
            chats: updatedChats,
            activeChat: updatedChat
          };
        });
      } else {
        // 送信失敗時のステータス更新
        setState(prev => {
          if (!prev.activeChat) return prev;

          const updatedMessages = prev.activeChat.messages.map(msg =>
            msg.id === newMessage.id ? { ...msg, status: 'error' as MessageStatus } : msg
          );

          const updatedChat = {
            ...prev.activeChat,
            messages: updatedMessages
          };

          const updatedChats = prev.chats.map(chat =>
            chat.id === updatedChat.id ? updatedChat : chat
          );

          saveChats(updatedChats);

          return {
            ...prev,
            chats: updatedChats,
            activeChat: updatedChat
          };
        });

        toast({
          title: 'メッセージ送信エラー',
          description: 'メッセージを送信できませんでした。接続を確認してください。',
          variant: 'destructive'
        });
      }

      return;
    }

    const newMessage: ChatMessage = {
      id: uuidv4(),
      content,
      sender: 'user',
      timestamp: new Date(),
      status: 'sending',
      attachments
    };

    setState(prev => {
      if (!prev.activeChat) return prev;

      const updatedChat = {
        ...prev.activeChat,
        messages: [...prev.activeChat.messages, newMessage],
        updatedAt: new Date()
      };

      const updatedChats = prev.chats.map(chat =>
        chat.id === updatedChat.id ? updatedChat : chat
      );

      saveChats(updatedChats);

      return {
        ...prev,
        chats: updatedChats,
        activeChat: updatedChat
      };
    });

    // WebSocketを使用してメッセージを送信
    const messageSent = ws.sendMessage('chat_message', {
      chatId: state.activeChat.id,
      message: {
        id: newMessage.id,
        content: newMessage.content,
        attachments: newMessage.attachments
      }
    });

    if (messageSent) {
      // 送信成功時のステータス更新
      setState(prev => {
        if (!prev.activeChat) return prev;

        const updatedMessages = prev.activeChat.messages.map(msg =>
          msg.id === newMessage.id ? { ...msg, status: 'sent' as MessageStatus } : msg
        );

        const updatedChat = {
          ...prev.activeChat,
          messages: updatedMessages
        };

        const updatedChats = prev.chats.map(chat =>
          chat.id === updatedChat.id ? updatedChat : chat
        );

        saveChats(updatedChats);

        return {
          ...prev,
          chats: updatedChats,
          activeChat: updatedChat
        };
      });
    } else {
      // 送信失敗時のステータス更新
      setState(prev => {
        if (!prev.activeChat) return prev;

        const updatedMessages = prev.activeChat.messages.map(msg =>
          msg.id === newMessage.id ? { ...msg, status: 'error' as MessageStatus } : msg
        );

        const updatedChat = {
          ...prev.activeChat,
          messages: updatedMessages
        };

        const updatedChats = prev.chats.map(chat =>
          chat.id === updatedChat.id ? updatedChat : chat
        );

        saveChats(updatedChats);

        return {
          ...prev,
          chats: updatedChats,
          activeChat: updatedChat
        };
      });

      toast({
        title: 'メッセージ送信エラー',
        description: 'メッセージを送信できませんでした。接続を確認してください。',
        variant: 'destructive'
      });
    }
  }, [state.activeChat, createChat, ws, saveChats, toast]);

  // AIからのメッセージを受信して追加
  const receiveAIMessage = useCallback((chatId: string, content: string, metadata?: any) => {
    const newMessage: ChatMessage = {
      id: uuidv4(),
      content,
      sender: 'ai',
      timestamp: new Date(),
      status: 'received',
      metadata
    };

    setState(prev => {
      const targetChat = prev.chats.find(chat => chat.id === chatId);

      if (!targetChat) return prev;

      const updatedChat = {
        ...targetChat,
        messages: [...targetChat.messages, newMessage],
        updatedAt: new Date()
      };

      const updatedChats = prev.chats.map(chat =>
        chat.id === chatId ? updatedChat : chat
      );

      saveChats(updatedChats);

      return {
        ...prev,
        chats: updatedChats,
        activeChat: prev.activeChat?.id === chatId ? updatedChat : prev.activeChat
      };
    });
  }, [saveChats]);

  // WebSocketからのメッセージを処理
  useEffect(() => {
    if (ws.status === 'connected' && ws.messages.length > 0) {
      const latestMessage = ws.getLatestMessage();

      if (latestMessage?.type === 'chat_response') {
        const { chatId, content, metadata } = latestMessage;
        receiveAIMessage(chatId, content, metadata);
      }
    }
  }, [ws.status, ws.messages, receiveAIMessage, ws.getLatestMessage]);

  // ユーザー認証時にチャット履歴を読み込む
  useEffect(() => {
    if (user) {
      loadChats();
    }
  }, [user, loadChats]);

  return {
    ...state,
    createChat,
    selectChat,
    deleteChat,
    renameChat,
    sendMessage,
    receiveAIMessage,
    wsStatus: ws.status
  };
};