import { AnalysisLayout } from "@/components/analysis-layout"
import { BayesianAnalysis } from "@/components/analysis/bayesian-analysis"

export default function BayesianPage() {
  return (
    <AnalysisLayout>
      <BayesianAnalysis />
    </AnalysisLayout>
  )
}