import React from 'react'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

interface GraphControlsProps {
  dashboardId: string
}

export function GraphControls({ dashboardId }: GraphControlsProps) {
  return (
    <div className="flex items-center justify-between gap-4 bg-background/95 p-4 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex items-center gap-4">
        <Select defaultValue="24h">
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="期間を選択" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="24h">24時間</SelectItem>
            <SelectItem value="7d">7日間</SelectItem>
            <SelectItem value="30d">30日間</SelectItem>
            <SelectItem value="custom">カスタム</SelectItem>
          </SelectContent>
        </Select>

        <Select defaultValue="all">
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="グラフタイプ" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">全て</SelectItem>
            <SelectItem value="users">ユーザー</SelectItem>
            <SelectItem value="revenue">収益</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="flex items-center gap-2">
        <Button variant="outline">リフレッシュ</Button>
        <Button>新規グラフ追加</Button>
      </div>
    </div>
  )
}