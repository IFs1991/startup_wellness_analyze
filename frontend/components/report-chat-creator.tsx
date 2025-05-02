import React, { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

export type ChatReportDraft = {
  company: string;
  messages: { role: "user" | "ai"; content: string }[];
};

type Props = {
  open: boolean;
  company: string;
  onComplete: (report: ChatReportDraft) => void;
  onClose: () => void;
};

export default function ReportChatCreator({ open, company, onComplete, onClose }: Props) {
  const [messages, setMessages] = useState<{ role: "user" | "ai"; content: string }[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;
    setMessages(msgs => [...msgs, { role: "user", content: input }]);
    setLoading(true);
    // 仮のAI応答
    setTimeout(() => {
      setMessages(msgs => [...msgs, { role: "ai", content: "AIによるレポート案: " + input }]);
      setLoading(false);
    }, 800);
    setInput("");
  };

  const handleComplete = () => {
    onComplete({ company, messages });
    setMessages([]);
    setInput("");
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>AIチャットでレポート作成（{company}）</DialogTitle>
        </DialogHeader>
        <div className="h-64 overflow-y-auto bg-gray-50 rounded p-2 mb-2 border">
          {messages.length === 0 && <div className="text-gray-400 text-sm">AIに相談しながらレポートを作成できます。</div>}
          {messages.map((msg, i) => (
            <div key={i} className={msg.role === "user" ? "text-right" : "text-left"}>
              <span className={msg.role === "user" ? "bg-blue-100 text-blue-800 px-2 py-1 rounded inline-block my-1" : "bg-gray-200 text-gray-700 px-2 py-1 rounded inline-block my-1"}>
                {msg.content}
              </span>
            </div>
          ))}
          {loading && <div className="text-xs text-gray-400">AIが応答中...</div>}
        </div>
        <div className="flex gap-2">
          <input
            className="input input-bordered flex-1"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter") handleSend(); }}
            placeholder="AIに質問や指示を入力..."
            disabled={loading}
          />
          <Button onClick={handleSend} disabled={loading || !input.trim()}>送信</Button>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>キャンセル</Button>
          <Button onClick={handleComplete} disabled={messages.length === 0}>レポート作成完了</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}