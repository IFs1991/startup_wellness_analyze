import React from 'react';
import { useAnalysis } from '../../hooks/useAnalysis';
import { VASChart } from '../charts/VASChart';
import { ProfitLossTable } from '../tables/ProfitLossTable';
import { DescriptiveStatsCard } from '../ui/DescriptiveStatsCard';
import { CorrelationMatrix } from '../charts/CorrelationMatrix';
import { TextAnalysisResults } from '../ui/TextAnalysisResults';
import { AISummary } from '../ui/AISummary';
import { LoadingSpinner } from '../ui/LoadingSpinner';
import { ErrorMessage } from '../ui/ErrorMessage';

interface AnalysisResultsProps {
  companyId: number;
}

export const AnalysisResults: React.FC<AnalysisResultsProps> = ({ companyId }) => {
  const {
    vasData,
    profitLossData,
    descriptiveStats,
    correlationAnalysis,
    textAnalysis,
    aiSummary,
    loading,
    error
  } = useAnalysis(companyId);

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error.message} />;

  // データが存在しない場合のチェック
  if (!vasData || !profitLossData || !descriptiveStats || !correlationAnalysis || !textAnalysis || !aiSummary) {
    return <ErrorMessage message="データの取得に失敗しました。" />;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
      {/* VASデータチャート */}
      <div className="col-span-2">
        <h2 className="text-xl font-bold mb-4">VASスケール分析</h2>
        <VASChart data={vasData} />
      </div>

      {/* 損益計算書データ */}
      <div className="col-span-2">
        <h2 className="text-xl font-bold mb-4">損益計算書分析</h2>
        <ProfitLossTable data={profitLossData} />
      </div>

      {/* 記述統計 */}
      <div>
        <h2 className="text-xl font-bold mb-4">記述統計</h2>
        <DescriptiveStatsCard stats={descriptiveStats} />
      </div>

      {/* 相関分析 */}
      <div>
        <h2 className="text-xl font-bold mb-4">相関分析</h2>
        <CorrelationMatrix data={correlationAnalysis} />
      </div>

      {/* テキスト分析 */}
      <div>
        <h2 className="text-xl font-bold mb-4">テキスト分析</h2>
        <TextAnalysisResults data={textAnalysis} />
      </div>

      {/* AI要約 */}
      <div>
        <h2 className="text-xl font-bold mb-4">AI分析要約</h2>
        <AISummary data={aiSummary} />
      </div>
    </div>
  );
};