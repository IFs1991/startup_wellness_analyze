import { AnalysisLayout } from "@/components/analysis-layout"
import { MontecarloAnalysis } from "@/components/analysis/montecarlo-analysis"

export default function MontecarloPage() {
  return (
    <AnalysisLayout>
      <MontecarloAnalysis />
    </AnalysisLayout>
  )
}