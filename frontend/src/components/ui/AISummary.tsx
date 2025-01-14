import React from 'react';

interface AISummaryData {
  summary: string;
  recommendations: string[];
  risk_factors: {
    factor: string;
    severity: 'low' | 'medium' | 'high';
  }[];
}

interface AISummaryProps {
  data: AISummaryData;
}

export const AISummary: React.FC<AISummaryProps> = ({ data }) => {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low':
        return 'bg-green-100 text-green-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'high':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow space-y-4">
      <div>
        <h3 className="font-bold mb-2">総合分析</h3>
        <p className="text-gray-700">{data.summary}</p>
      </div>

      <div>
        <h3 className="font-bold mb-2">推奨アクション</h3>
        <ul className="list-disc list-inside space-y-1">
          {data.recommendations.map((recommendation, index) => (
            <li key={index} className="text-gray-700">
              {recommendation}
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h3 className="font-bold mb-2">リスク要因</h3>
        <div className="space-y-2">
          {data.risk_factors.map((risk, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-2 rounded"
            >
              <span>{risk.factor}</span>
              <span
                className={`px-2 py-1 rounded-full text-sm ${getSeverityColor(
                  risk.severity
                )}`}
              >
                {risk.severity === 'low' && '低'}
                {risk.severity === 'medium' && '中'}
                {risk.severity === 'high' && '高'}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};