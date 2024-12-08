import { DashboardCard } from "@/components/dashboard/dashboard-card"
import { DashboardGrid } from "@/components/dashboard/dashboard-grid"
import { DashboardHeader } from "@/components/dashboard/dashboard-header"

export default function DashboardPage() {
  return (
    <div className="flex flex-col gap-4 p-4">
      <DashboardHeader />
      <DashboardGrid>
        {/* ダッシュボードウィジェットをマッピング */}
      </DashboardGrid>
    </div>
  )
}