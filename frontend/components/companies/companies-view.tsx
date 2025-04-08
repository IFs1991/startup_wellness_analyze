"use client"

import type React from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Plus, Search, SlidersHorizontal } from "lucide-react"
import { CompaniesTable } from "@/components/companies/companies-table"
import { CompaniesGrid } from "@/components/companies/companies-grid"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { memo, useState, useCallback } from "react"
import { companyFilters } from "@/lib/constants"

export const CompaniesView = memo(function CompaniesView() {
  const [searchQuery, setSearchQuery] = useState("")
  const [filter, setFilter] = useState("all")

  const handleSearchChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value)
  }, [])

  const handleFilterChange = useCallback((value: string) => {
    setFilter(value)
  }, [])

  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">企業管理</h2>
          <p className="text-sm text-text-secondary">企業データの管理と分析</p>
        </div>
        <div className="flex items-center gap-2">
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            企業を追加
          </Button>
        </div>
      </div>

      <div className="mb-4 flex flex-col gap-4 sm:flex-row">
        <div className="relative flex-1">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-text-muted" />
          <Input placeholder="企業を検索..." className="pl-8" value={searchQuery} onChange={handleSearchChange} />
        </div>
        <div className="flex gap-2">
          <Select value={filter} onValueChange={handleFilterChange}>
            <SelectTrigger className="w-[120px]">
              <SelectValue placeholder="フィルター" />
            </SelectTrigger>
            <SelectContent>
              {companyFilters.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon">
            <SlidersHorizontal className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Tabs defaultValue="table">
        <div className="mb-4 flex items-center justify-between">
          <TabsList>
            <TabsTrigger value="table">テーブル</TabsTrigger>
            <TabsTrigger value="grid">グリッド</TabsTrigger>
          </TabsList>
          <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            エクスポート
          </Button>
        </div>

        <TabsContent value="table" className="mt-0">
          <CompaniesTable searchQuery={searchQuery} filter={filter} />
        </TabsContent>

        <TabsContent value="grid" className="mt-0">
          <CompaniesGrid searchQuery={searchQuery} filter={filter} />
        </TabsContent>
      </Tabs>
    </div>
  )
})

