import { AnalysisLayout } from "@/components/analysis-layout"
import { CausalInferenceAnalysis } from "@/components/analysis/causal-inference-analysis"

export default function CausalInferencePage() {
  return (
    <AnalysisLayout>
      <CausalInferenceAnalysis />
    </AnalysisLayout>
  )
}