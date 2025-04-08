"use client"

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Star, TrendingDown, TrendingUp } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { memo, useMemo } from "react"
import { companies } from "@/lib/constants"
import { getWellnessScoreColor, getWellnessScoreProgressColor } from "@/lib/utils"

interface CompaniesTableProps {
  searchQuery: string
  filter: string
}

export const CompaniesTable = memo(function CompaniesTable({ searchQuery, filter }: CompaniesTableProps) {
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
    <div className="rounded-md border border-background-lighter">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[50px]"></TableHead>
            <TableHead>企業名</TableHead>
            <TableHead>業種</TableHead>
            <TableHead>ウェルネス</TableHead>
            <TableHead>成長率</TableHead>
            <TableHead className="text-right">アクション</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {filteredCompanies.length > 0 ? (
            filteredCompanies.map((company) => (
              <TableRow key={company.id}>
                <TableCell>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <Star className={`h-4 w-4 ${company.starred ? "fill-accent text-accent" : "text-text-muted"}`} />
                  </Button>
                </TableCell>
                <TableCell className="font-medium">{company.name}</TableCell>
                <TableCell>
                  <Badge variant="outline">{company.industry}</Badge>
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span className={getWellnessScoreColor(company.wellnessScore)}>{company.wellnessScore}</span>
                    <Progress
                      value={company.wellnessScore}
                      className="h-2 w-16"
                      indicatorClassName={getWellnessScoreProgressColor(company.wellnessScore)}
                    />
                  </div>
                </TableCell>
                <TableCell>{getGrowthRateElement(company.growthRate, company.trend)}</TableCell>
                <TableCell className="text-right">
                  <Button variant="outline" size="sm">
                    詳細
                  </Button>
                </TableCell>
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={6} className="h-24 text-center">
                該当する企業がありません
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  )
})

