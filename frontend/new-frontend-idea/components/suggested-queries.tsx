"use client"

import { Button } from "@/components/ui/button"
import { memo } from "react"
import { suggestedQueries } from "@/lib/constants"

interface SuggestedQueriesProps {
  onSelectQuery: (query: string) => void
}

export const SuggestedQueries = memo(function SuggestedQueries({ onSelectQuery }: SuggestedQueriesProps) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-text-secondary">おすすめの質問:</h3>
      <div className="flex flex-wrap gap-2">
        {suggestedQueries.map((query, index) => (
          <Button
            key={index}
            variant="outline"
            size="sm"
            className="bg-background-light text-text-secondary"
            onClick={() => onSelectQuery(query)}
          >
            {query}
          </Button>
        ))}
      </div>
    </div>
  )
})

