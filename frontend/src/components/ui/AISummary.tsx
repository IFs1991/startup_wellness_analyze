import React from 'react';

interface AISummaryData {
  summary: string;
  recommendations: string[];
  key_insights: string[];
}

interface AISummaryProps {
  data: AISummaryData;
}

export const AISummary: React.FC<AISummaryProps> = ({ data }) => {
  return (
    <div className="bg-white shadow rounded-lg p-6">
      <div className="space-y-6">
        {/* 要約 */}
        <div>
          <h3 className="text-lg font-semibold mb-2">分析要約</h3>
          <p className="text-gray-700">{data.summary}</p>
        </div>

        {/* 主要な洞察 */}
        <div>
          <h3 className="text-lg font-semibold mb-2">主要な洞察</h3>
          <ul className="list-disc list-inside space-y-2">
            {data.key_insights.map((insight, index) => (
              <li key={index} className="text-gray-700">{insight}</li>
            ))}
          </ul>
        </div>

        {/* 推奨事項 */}
        <div>
          <h3 className="text-lg font-semibold mb-2">推奨事項</h3>
          <ul className="list-disc list-inside space-y-2">
            {data.recommendations.map((recommendation, index) => (
              <li key={index} className="text-gray-700">{recommendation}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};