import React from 'react'
import { GraphCard } from './graph-card'

interface GraphData {
  id: string
  title: string
  data: Array<any>
  xKey: string
  yKey: string
}

interface GraphGridProps {
  dashboardId: string
}

export function GraphGrid({ dashboardId }: GraphGridProps) {
  // この部分は実際のデータフェッチロジックに置き換える必要があります
  const sampleData: GraphData[] = [
    {
      id: '1',
      title: 'Daily Active Users',
      data: [
        { date: '2024-01-01', value: 100 },
        { date: '2024-01-02', value: 120 },
        { date: '2024-01-03', value: 150 },
      ],
      xKey: 'date',
      yKey: 'value'
    },
    {
      id: '2',
      title: 'Revenue',
      data: [
        { date: '2024-01-01', value: 1000 },
        { date: '2024-01-02', value: 1200 },
        { date: '2024-01-03', value: 1500 },
      ],
      xKey: 'date',
      yKey: 'value'
    }
  ]

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {sampleData.map((graph) => (
        <GraphCard
          key={graph.id}
          title={graph.title}
          data={graph.data}
          xKey={graph.xKey}
          yKey={graph.yKey}
          className="w-full"
        />
      ))}
    </div>
  )
}