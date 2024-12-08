import { Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

export function CreateDashboardButton() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          新規ダッシュボード
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>新規ダッシュボードの作成</DialogTitle>
          <DialogDescription>
            新しいダッシュボードを作成します。必要な情報を入力してください。
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          {/* ここにフォームコンポーネントを追加予定 */}
        </div>
      </DialogContent>
    </Dialog>
  )
}