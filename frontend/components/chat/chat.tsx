"use client";

import { useState, useEffect } from 'react';
import { ChatContainer } from './chat-container';
import { ChatInput } from './chat-input';
import { useChatState } from '@/hooks/useChatState';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Trash2, RotateCcw, WifiOff } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Insight } from '@/lib/ai-insights-generator';
import { FileAttachment } from './file-attachment';
import { Loader2 } from 'lucide-react';
import { AlertCircle } from 'lucide-react';

export function Chat() {
  const {
    messages,
    addUserMessage,
    addAssistantMessage,
    updateMessageStatus,
    clearMessages,
    isProcessing,
    setIsProcessing,
    error,
    connectionStatus,
    reconnect,
    retryMessage
  } = useChatState();

  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // エラー状態の管理
  useEffect(() => {
    if (error) {
      setErrorMessage(error.message);
    } else {
      setErrorMessage(null);
    }
  }, [error]);

  // 接続状態の管理
  useEffect(() => {
    if (connectionStatus === 'error' || connectionStatus === 'disconnected') {
      setErrorMessage('サーバーとの接続が切断されました。再接続を試みてください。');
    }
  }, [connectionStatus]);

  const handleSendMessage = async (content: string, attachments?: FileAttachment[]) => {
    // ファイル添付がある場合は、まずアップロードする
    let uploadedAttachments = attachments;

    const messageId = addUserMessage(content, attachments);
    setIsProcessing(true);
    setErrorMessage(null);

    try {
      // FormDataを使用してファイルと一緒にメッセージを送信する準備
      const formData = new FormData();
      formData.append('message', content);

      // ファイル添付がある場合はFormDataに追加
      if (attachments && attachments.length > 0) {
        // メタデータJSONを追加
        const attachmentsMetadata = attachments.map(att => ({
          id: att.id,
          name: att.file.name,
          type: att.file.type,
          size: att.file.size
        }));
        formData.append('attachmentsMetadata', JSON.stringify(attachmentsMetadata));

        // 各ファイルを追加
        attachments.forEach(attachment => {
          formData.append('files', attachment.file, attachment.file.name);
        });
      }

      // ファイル添付がある場合はFormDataを使用してPOSTリクエスト
      if (attachments && attachments.length > 0) {
        const response = await fetch('/api/chat/with-attachments', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('APIレスポンスエラー: ' + response.status);
        }

        const data = await response.json();
        updateMessageStatus(messageId, 'sent');

        // 分析結果があれば含める
        const analysis = data.analysis ? {
          title: data.analysis.title,
          description: data.analysis.description,
          insights: data.analysis.insights as Insight[],
          analysisType: data.analysis.analysisType
        } : undefined;

        addAssistantMessage(data.response, analysis);
      } else {
        // 添付ファイルなしの場合は通常のJSON POSTリクエスト
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: content
          })
        });

        if (!response.ok) {
          throw new Error('APIレスポンスエラー: ' + response.status);
        }

        const data = await response.json();
        updateMessageStatus(messageId, 'sent');

        // 分析結果があれば含める
        const analysis = data.analysis ? {
          title: data.analysis.title,
          description: data.analysis.description,
          insights: data.analysis.insights as Insight[],
          analysisType: data.analysis.analysisType
        } : undefined;

        addAssistantMessage(data.response, analysis);
      }

      setIsProcessing(false);
    } catch (err) {
      console.error('メッセージ送信エラー:', err);
      updateMessageStatus(messageId, 'error');
      setErrorMessage('メッセージの送信に失敗しました。ネットワーク接続を確認してください。');
      setIsProcessing(false);
    }
  };

  // 分析リクエストの処理
  const handleAnalysisRequest = async (analysisType: string, parameters?: Record<string, any>) => {
    // 分析リクエストメッセージを追加
    const requestMessage = `${analysisType}分析を実行しています...\n${
      parameters ? `パラメータ: ${JSON.stringify(parameters, null, 2)}` : ''
    }`;

    const messageId = addUserMessage(requestMessage);
    setIsProcessing(true);
    setErrorMessage(null);

    try {
      // 分析リクエストAPIを呼び出す
      const response = await fetch('/api/analysis/request', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysisType,
          parameters
        })
      });

      if (!response.ok) {
        throw new Error('APIレスポンスエラー: ' + response.status);
      }

      const data = await response.json();
      updateMessageStatus(messageId, 'sent');

      // 分析結果を含めた応答を追加
      const analysis = {
        title: data.title || `${analysisType}分析の結果`,
        description: data.description,
        insights: data.insights,
        analysisType: analysisType
      };

      addAssistantMessage(data.response || '分析結果を表示しています', analysis);
    } catch (error) {
      console.error('分析リクエストエラー:', error);
      updateMessageStatus(messageId, 'error');
      setErrorMessage('分析リクエストの処理中にエラーが発生しました。');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClearChat = () => {
    clearMessages();
    setErrorMessage(null);
  };

  const handleRetryConnection = () => {
    reconnect();
    setErrorMessage(null);
  };

  const handleRetryMessage = (messageId: string) => {
    retryMessage(messageId);
  };

  const isConnected = connectionStatus === 'connected';
  const isConnecting = connectionStatus === 'connecting' || connectionStatus === 'reconnecting';

  return (
    <Card className="w-full max-w-3xl h-[600px] flex flex-col">
      <CardHeader className="p-4 flex-row justify-between items-center">
        <CardTitle className="flex items-center gap-2">
          ウェルネスアシスタント
          {isConnected && <span className="h-2 w-2 rounded-full bg-green-500" title="接続済み" />}
          {isConnecting && <span className="h-2 w-2 rounded-full bg-yellow-500 animate-pulse" title="接続中..." />}
          {(connectionStatus === 'error' || connectionStatus === 'disconnected') &&
            <span className="h-2 w-2 rounded-full bg-red-500" title="接続エラー" />
          }
        </CardTitle>
        <div className="flex gap-2">
          {(connectionStatus === 'error' || connectionStatus === 'disconnected') && (
            <Button
              variant="outline"
              size="icon"
              onClick={handleRetryConnection}
              title="再接続"
            >
              <RotateCcw className="h-4 w-4" />
              <span className="sr-only">再接続</span>
            </Button>
          )}
          <Button
            variant="outline"
            size="icon"
            onClick={handleClearChat}
            disabled={messages.length <= 1 || isProcessing}
            title="チャット履歴を消去"
          >
            <Trash2 className="h-4 w-4" />
            <span className="sr-only">チャットをクリア</span>
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex-1 overflow-hidden p-0">
        <ChatContainer
          messages={messages}
          className="h-full"
          onRetry={handleRetryMessage}
        />
      </CardContent>

      <CardFooter className="p-4 pt-2 flex-col">
        {errorMessage && (
          <Alert variant="destructive" className="mb-3 py-2">
            <WifiOff className="h-4 w-4 mr-2" />
            <AlertTitle className="text-sm">エラー</AlertTitle>
            <AlertDescription className="text-xs">
              {errorMessage}
              {(connectionStatus === 'error' || connectionStatus === 'disconnected') && (
                <Button
                  variant="outline"
                  size="sm"
                  className="ml-2 h-6 text-xs"
                  onClick={handleRetryConnection}
                >
                  再接続
                </Button>
              )}
            </AlertDescription>
          </Alert>
        )}
        <ChatInput
          onSendMessage={handleSendMessage}
          onRequestAnalysis={handleAnalysisRequest}
          isDisabled={connectionStatus !== 'connected'}
          isProcessing={isProcessing}
          className="w-full"
          placeholder={
            isConnecting
              ? '接続中...'
              : connectionStatus === 'error' || connectionStatus === 'disconnected'
              ? '接続エラー。再接続してください'
              : 'メッセージを入力...'
          }
        />
      </CardFooter>
    </Card>
  );
}