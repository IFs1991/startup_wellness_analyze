import { useState, useEffect } from 'react';
import { api } from '../lib/api/client';

export const useAnalysis = (companyId: number) => {
  const [vasData, setVasData] = useState(null);
  const [profitLossData, setProfitLossData] = useState(null);
  const [descriptiveStats, setDescriptiveStats] = useState(null);
  const [correlationAnalysis, setCorrelationAnalysis] = useState(null);
  const [textAnalysis, setTextAnalysis] = useState(null);
  const [aiSummary, setAiSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchAnalysisData = async () => {
      try {
        setLoading(true);
        const [
          vasResponse,
          profitLossResponse,
          statsResponse,
          correlationResponse,
          textResponse,
          aiResponse
        ] = await Promise.all([
          api.analysis.getVasData(companyId),
          api.analysis.getProfitLossData(companyId),
          api.analysis.getDescriptiveStats(companyId),
          api.analysis.getCorrelationAnalysis(companyId),
          api.analysis.getTextAnalysis(companyId),
          api.analysis.getAiSummary(companyId)
        ]);

        setVasData(vasResponse.data);
        setProfitLossData(profitLossResponse.data);
        setDescriptiveStats(statsResponse.data);
        setCorrelationAnalysis(correlationResponse.data);
        setTextAnalysis(textResponse.data);
        setAiSummary(aiResponse.data);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('分析データの取得に失敗しました'));
      } finally {
        setLoading(false);
      }
    };

    if (companyId) {
      fetchAnalysisData();
    }
  }, [companyId]);

  return {
    vasData,
    profitLossData,
    descriptiveStats,
    correlationAnalysis,
    textAnalysis,
    aiSummary,
    loading,
    error
  };
};