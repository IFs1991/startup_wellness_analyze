"use client"

import { useEffect, useState, memo } from "react"

interface TimeSeriesChartProps {
  type: "wellness" | "growth" | "combined" | "comparison"
}

export const TimeSeriesChart = memo(function TimeSeriesChart({ type }: TimeSeriesChartProps) {
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  // 実際のアプリケーションでは、ここでチャートライブラリ（recharts, visx, d3など）を使用してデータを可視化します
  // このデモでは、プレースホルダーを表示します

  const getChartTitle = () => {
    switch (type) {
      case "wellness":
        return "ウェルネススコア推移"
      case "growth":
        return "成長率推移"
      case "combined":
        return "複合指標推移"
      case "comparison":
        return "企業比較"
    }
  }

  if (!isClient) {
    return <div className="h-[400px] rounded-md bg-background-main"></div>
  }

  return (
    <div className="h-[400px] rounded-md bg-background-main p-4">
      <div className="flex h-full flex-col items-center justify-center">
        <div className="text-lg font-medium">{getChartTitle()}</div>
        <div className="mt-2 text-sm text-text-secondary">
          {type === "wellness" && "企業ごとのウェルネススコアの時間的変化を表示します"}
          {type === "growth" && "企業ごとの成長率の時間的変化を表示します"}
          {type === "combined" && "ウェルネススコアと成長率の複合指標を表示します"}
          {type === "comparison" && "選択した企業間のウェルネススコア比較を表示します"}
        </div>
        <div className="mt-8 text-text-muted">実際のアプリケーションでは、ここに時系列チャートが表示されます</div>
      </div>
    </div>
  )
})

