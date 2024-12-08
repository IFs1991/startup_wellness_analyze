import { DashboardCard } from "./dashboard-card"

const DEMO_DATA = [
  {
    id: "1",
    title: "月間アクティブユーザー",
    value: "2,842",
    description: "前月比 +20%",
    chartData: [10, 20, 30, 40, 50, 40, 30],
  },
  {
    id: "2",
    title: "平均セッション時間",
    value: "24分",
    description: "前月比 +5%",
    chartData: [20, 30, 25, 35, 45, 40, 50],
  },
  {
    id: "3",
    title: "コンバージョン率",
    value: "3.2%",
    description: "前月比 +1.2%",
    chartData: [15, 25, 35, 45, 40, 50, 45],
  },
  {
    id: "4",
    title: "総収益",
    value: "¥1,234,567",
    description: "前月比 +15%",
    chartData: [30, 40, 45, 50, 55, 60, 65],
  },
]

export function DashboardGrid() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {DEMO_DATA.map((item) => (
        <DashboardCard
          key={item.id}
          title={item.title}
          value={item.value}
          description={item.description}
          chartData={item.chartData}
        />
      ))}
    </div>
  )
}