import React from 'react'

// DashboardCardの必須プロパティを定義するインターフェース
interface DashboardCardProps {
  title: string      // カードのタイトル (例: "月間アクティブユーザー")
  value: string      // 表示する主要な値 (例: "2,842")
  description: string // 補足説明 (例: "前月比 +20%")
  chartData: number[] // グラフ用のデータ配列
}

// デフォルトエクスポートとして定義
export default function DashboardCard({
  title,
  value,
  description,
  chartData
}: DashboardCardProps) {
  return (
    <div className="w-full h-full rounded-lg border p-4 bg-white">
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-gray-500">{title}</h3>
        <p className="text-2xl font-bold">{value}</p>
        <p className="text-sm text-gray-600">{description}</p>
        {/* チャートの実装部分はここに配置 */}
        <div className="h-16">
          {/* グラフの実装は後ほど必要に応じて追加 */}
        </div>
      </div>
    </div>
  )
}

// 名前付きエクスポートも提供
export { DashboardCard }