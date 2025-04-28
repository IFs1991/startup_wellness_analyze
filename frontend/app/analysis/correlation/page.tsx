import { AnalysisLayout } from "@/components/analysis-layout"
import { CorrelationAnalysis } from "@/components/analysis/correlation-analysis"

export default function CorrelationPage() {
  return (
    <AnalysisLayout>
      <CorrelationAnalysis />
    </AnalysisLayout>
  )
}