"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { TrendingUp, TrendingDown, AlertTriangle, Award } from "lucide-react"
import { memo } from "react"

export const DashboardContent = memo(function DashboardContent() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">平均ウェルネススコア</CardTitle>
          <CardDescription>全企業の平均値</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">72.4</div>
          <div className="mt-2 flex items-center text-xs text-secondary">
            <TrendingUp className="mr-1 h-3 w-3" />
            <span>前月比 +2.3%</span>
          </div>
          <Progress value={72.4} className="mt-2" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">平均成長率</CardTitle>
          <CardDescription>全企業の平均値</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">8.7%</div>
          <div className="mt-2 flex items-center text-xs text-warning">
            <TrendingDown className="mr-1 h-3 w-3" />
            <span>前月比 -1.2%</span>
          </div>
          <Progress value={8.7} max={20} className="mt-2" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">リスク企業数</CardTitle>
          <CardDescription>ウェルネススコア50未満</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">12</div>
          <div className="mt-2 flex items-center text-xs text-warning">
            <AlertTriangle className="mr-1 h-3 w-3" />
            <span>前月比 +2社</span>
          </div>
          <Progress value={12} max={100} className="mt-2" />
        </CardContent>
      </Card>

      <Card className="md:col-span-2">
        <CardHeader>
          <CardTitle>ウェルネススコア推移</CardTitle>
          <CardDescription>過去3ヶ月間の平均値推移</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[200px] rounded-md bg-background-main p-4 text-center text-text-muted">
            時系列チャートがここに表示されます
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>トップパフォーマー</CardTitle>
          <CardDescription>ウェルネススコア上位企業</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Award className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">テックスタート社</span>
              </div>
              <span className="text-sm">92.5</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Award className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">フューチャーラボ</span>
              </div>
              <span className="text-sm">89.3</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Award className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">イノベーションテック</span>
              </div>
              <span className="text-sm">87.1</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
})

export default { DashboardContent }

