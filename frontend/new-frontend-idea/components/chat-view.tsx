"use client"

import type React from "react"

import { useState, useCallback, memo } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Send } from "lucide-react"
import { SuggestedQueries } from "@/components/suggested-queries"
import type { ChatMessage } from "@/types"

export const ChatView = memo(function ChatView() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "こんにちは、スタートアップウェルネスアナライザーへようこそ。企業分析やポートフォリオについて質問してください。",
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState("")

  const handleSendMessage = useCallback(() => {
    if (!input.trim()) return

    // Add user message
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")

    // Simulate assistant response (in a real app, this would be an API call)
    setTimeout(() => {
      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content:
          "申し訳ありませんが、現在デモモードのため、実際の分析結果を提供できません。実際のアプリケーションでは、ここに企業分析の結果が表示されます。",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, assistantMessage])
    }, 1000)
  }, [input])

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
  }, [])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault()
        handleSendMessage()
      }
    },
    [handleSendMessage],
  )

  const handleSelectQuery = useCallback((query: string) => {
    setInput(query)
  }, [])

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto p-4">
        <div className="mx-auto max-w-3xl space-y-4">
          {messages.map((message) => (
            <Card
              key={message.id}
              className={`p-4 ${message.role === "user" ? "ml-12 bg-background-light" : "mr-12 bg-background-lighter"}`}
            >
              {message.content}
            </Card>
          ))}
          {messages.length === 1 && <SuggestedQueries onSelectQuery={handleSelectQuery} />}
        </div>
      </div>
      <div className="border-t border-background-lighter bg-background-light p-4">
        <div className="mx-auto flex max-w-3xl items-center gap-2">
          <Input
            value={input}
            onChange={handleInputChange}
            placeholder="質問を入力してください..."
            className="flex-1"
            onKeyDown={handleKeyDown}
          />
          <Button onClick={handleSendMessage} size="icon">
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
})

export default { ChatView }

