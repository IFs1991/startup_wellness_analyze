"use client"

import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Star, TrendingDown, TrendingUp } from "lucide-react"
import { memo, useMemo } from "react"
import { companies } from "@/lib/constants"
import { getWellnessScoreColor, getWellnessScoreProgressColor } from "@/lib/utils"

interface CompaniesGridProps {
  searchQuery: string
  filter: string
}

export const CompaniesGrid = memo(function CompaniesGrid({ searchQuery, filter }: CompaniesGridProps) {
  // 検索とフィルタリングを適用した企業リスト
  const filteredCompanies = useMemo(() => {
    let result = [...companies]

    // 検索クエリによるフィルタリング
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      result = result.filter(
        (company) => company.name.toLowerCase().includes(query) || company.industry.toLowerCase().includes(query),
      )
    }

    // フィルターによるフィルタリング
    if (filter !== "all") {
      if (filter === "starred") {
        result = result.filter((company) => company.starred)
      } else if (filter === "trending") {
        result = result.filter((company) => company.trend === "up")
      } else if (filter === "risk") {
        result = result.filter((company) => company.wellnessScore < 60)
      }
    }

    return result
  }, [searchQuery, filter])

  const getGrowthRateElement = (rate: number, trend: string) => {
    if (trend === "up") {
      return (
        <div className="flex items-center text-secondary">
          <TrendingUp className="mr-1 h-4 w-4" />
          <span>{rate}%</span>
        </div>
      )
    }
    return (
      <div className="flex items-center text-warning">
        <TrendingDown className="mr-1 h-4 w-4" />
        <span>{rate}%</span>
      </div>
    )
  }

  return (
    <>
      {filteredCompanies.length > 0 ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filteredCompanies.map((company) => (
            <Card key={company.id}>
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                  <Badge variant="outline">{company.industry}</Badge>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <Star className={`h-4 w-4 ${company.starred ? "fill-accent text-accent" : "text-text-muted"}`} />
                  </Button>
                </div>
                <h3 className="text-lg font-semibold">{company.name}</h3>
              </CardHeader>
              <CardContent className="pb-2">
                <div className="space-y-4">
                  <div>
                    <div className="mb-1 flex items-center justify-between">
                      <span className="text-sm text-text-secondary">ウェルネススコア</span>
                      <span className={`font-medium ${getWellnessScoreColor(company.wellnessScore)}`}>
                        {company.wellnessScore}
                      </span>
                    </div>
                    <Progress
                      value={company.wellnessScore}
                      className="h-2"
                      indicatorClassName={getWellnessScoreProgressColor(company.wellnessScore)}
                    />
                  </div>
                  <div>
                    <div className="mb-1 flex items-center justify-between">
                      <span className="text-sm text-text-secondary">成長率</span>
                      {getGrowthRateElement(company.growthRate, company.trend)}
                    </div>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button variant="outline" className="w-full">
                  詳細を表示
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      ) : (
        <div className="flex h-40 items-center justify-center rounded-md border border-background-lighter">
          <p className="text-text-muted">該当する企業がありません</p>
        </div>
      )}
    </>
  )
})

