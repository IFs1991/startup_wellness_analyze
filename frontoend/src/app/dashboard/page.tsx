import { Suspense } from "react"
import { notFound } from "next/navigation"
import { auth } from "@/lib/firebase"
import { GraphGrid } from "@/components/graphs/graph-grid"
import { GraphControls } from "@/components/graphs/graph-controls"
import { Skeleton } from "@/components/ui/skeleton"
import {
  PageHeader,
  PageHeaderDescription,
  PageHeaderHeading,
} from "@/components/shared/page-header"

interface DashboardPageProps {
  params: {
    id: string
  }
}

async function getDashboard(id: string) {
  try {
    const token = await auth.currentUser?.getIdToken()
    // TODO: APIからダッシュボードデータを取得
    // この部分は実際のAPIエンドポイントに合わせて実装する必要があります
    return {
      id,
      title: "サンプルダッシュボード",
      description: "これはサンプルのダッシュボードです",
      // 他の必要なデータ
    }
  } catch (error) {
    return null
  }
}

export default async function DashboardDetailPage({ params }: DashboardPageProps) {
  const dashboard = await getDashboard(params.id)

  if (!dashboard) {
    notFound()
  }

  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
      <PageHeader>
        <PageHeaderHeading>{dashboard.title}</PageHeaderHeading>
        <PageHeaderDescription>
          {dashboard.description}
        </PageHeaderDescription>
      </PageHeader>

      <div className="space-y-4">
        <GraphControls dashboardId={dashboard.id} />

        <Suspense fallback={<GraphSkeleton />}>
          <GraphGrid dashboardId={dashboard.id} />
        </Suspense>
      </div>
    </div>
  )
}

function GraphSkeleton() {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {Array.from({ length: 4 }).map((_, index) => (
        <Skeleton
          key={`skeleton-${index}`}
          className="aspect-square w-full rounded-xl"
        />
      ))}
    </div>
  )
}

export async function generateMetadata({ params }: DashboardPageProps) {
  const dashboard = await getDashboard(params.id)

  if (!dashboard) {
    return {
      title: "ダッシュボードが見つかりません",
    }
  }

  return {
    title: dashboard.title,
    description: dashboard.description,
  }
}