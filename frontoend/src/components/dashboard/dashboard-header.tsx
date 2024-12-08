import { Search } from "lucide-react"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export function DashboardHeader() {
  return (
    <div className="flex items-center gap-4">
      <div className="relative w-64">
        <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input placeholder="検索..." className="pl-8" />
      </div>
      <Select defaultValue="all">
        <SelectTrigger className="w-40">
          <SelectValue placeholder="カテゴリー" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">すべて</SelectItem>
          <SelectItem value="analytics">分析</SelectItem>
          <SelectItem value="reports">レポート</SelectItem>
          <SelectItem value="settings">設定</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}