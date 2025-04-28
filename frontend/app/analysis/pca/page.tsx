import { AnalysisLayout } from "@/components/analysis-layout"
import { PcaAnalysis } from "@/components/analysis/pca-analysis"

export default function PcaPage() {
  return (
    <AnalysisLayout>
      <PcaAnalysis />
    </AnalysisLayout>
  )
}