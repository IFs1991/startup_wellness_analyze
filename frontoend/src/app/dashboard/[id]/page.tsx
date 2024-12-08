import { Suspense } from "react"
import { DashboardHeader } from "@/components/dashboard/dashboard-header"
import { DashboardGrid } from "@/components/dashboard/dashboard-grid"
import { CreateDashboardButton } from "@/components/dashboard/create-dashboard-button"
import { Skeleton } from "@/components/ui/skeleton"
import {
  PageHeader,
  PageHeaderDescription,
  PageHeaderHeading,
} from "@/components/shared/page-header"

export default async function DashboardPage() {
  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
      <PageHeader>
        <PageHeaderHeading>ダッシュボード</PageHeaderHeading>
        <PageHeaderDescription>
          データ分析とビジュアライゼーションのダッシュボード
        </PageHeaderDescription>
      </PageHeader>
      <div className="flex items-center justify-between">
        <DashboardHeader />
        <CreateDashboardButton />
      </div>
      <Suspense fallback={<DashboardSkeleton />}>
        <DashboardGrid />
      </Suspense>
    </div>
  )
}

function DashboardSkeleton() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: 4 }).map((_, index) => (
        <Skeleton
          key={`skeleton-${index}`}
          className="h-48 w-full rounded-xl"
        />
      ))}
    </div>
  )
}