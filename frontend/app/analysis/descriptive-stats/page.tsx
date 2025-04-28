import { AnalysisLayout } from "@/components/analysis-layout"
import { DescriptiveStatsAnalysis } from "@/components/analysis/descriptive-stats-analysis"

export default function DescriptiveStatsPage() {
  return (
    <AnalysisLayout>
      <DescriptiveStatsAnalysis />
    </AnalysisLayout>
  )
}