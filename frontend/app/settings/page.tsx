"use client"

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { useState } from "react"
import { AnalysisLayout } from "@/components/analysis-layout"

interface User {
  id: number
  name: string
  email: string
}

const initialUsers: User[] = [
  { id: 1, name: "山田 太郎", email: "taro@example.com" },
  { id: 2, name: "佐藤 花子", email: "hanako@example.com" },
]

export default function SettingsPage() {
  const [users, setUsers] = useState<User[]>(initialUsers)
  const [newUser, setNewUser] = useState({ name: "", email: "" })

  const handleAddUser = () => {
    if (!newUser.name || !newUser.email) return
    setUsers([
      ...users,
      { id: Date.now(), name: newUser.name, email: newUser.email },
    ])
    setNewUser({ name: "", email: "" })
  }

  const handleDeleteUser = (id: number) => {
    setUsers(users.filter((u) => u.id !== id))
  }

  return (
    <AnalysisLayout>
      <div className="max-w-3xl mx-auto py-10 px-4">
        <h1 className="text-2xl font-bold mb-6">設定</h1>
        <Tabs defaultValue="users">
          <TabsList>
            <TabsTrigger value="users">ユーザー管理</TabsTrigger>
            {/* 今後の拡張用タブ */}
          </TabsList>
          <TabsContent value="users">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-2">ユーザー一覧</h2>
              <table className="w-full border text-sm">
                <thead>
                  <tr className="bg-muted">
                    <th className="p-2 border">名前</th>
                    <th className="p-2 border">メールアドレス</th>
                    <th className="p-2 border">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((user) => (
                    <tr key={user.id}>
                      <td className="p-2 border">{user.name}</td>
                      <td className="p-2 border">{user.email}</td>
                      <td className="p-2 border text-center">
                        <Button size="sm" variant="destructive" onClick={() => handleDeleteUser(user.id)}>
                          削除
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-8">
              <h2 className="text-lg font-semibold mb-2">ユーザー追加</h2>
              <div className="flex gap-2">
                <input
                  className="border rounded px-2 py-1 flex-1"
                  placeholder="名前"
                  value={newUser.name}
                  onChange={e => setNewUser({ ...newUser, name: e.target.value })}
                />
                <input
                  className="border rounded px-2 py-1 flex-1"
                  placeholder="メールアドレス"
                  value={newUser.email}
                  onChange={e => setNewUser({ ...newUser, email: e.target.value })}
                />
                <Button onClick={handleAddUser}>追加</Button>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </AnalysisLayout>
  )
}