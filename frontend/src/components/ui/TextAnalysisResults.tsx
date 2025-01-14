import React from 'react';

interface TextAnalysisData {
  sentiment_score: number;
  key_phrases: string[];
  topics: {
    topic: string;
    confidence: number;
  }[];
}

interface TextAnalysisResultsProps {
  data: TextAnalysisData;
}

export const TextAnalysisResults: React.FC<TextAnalysisResultsProps> = ({ data }) => {
  const getSentimentColor = (score: number) => {
    if (score > 0.5) return 'text-green-600';
    if (score < -0.5) return 'text-red-600';
    return 'text-yellow-600';
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow space-y-4">
      <div>
        <h3 className="font-bold mb-2">感情分析スコア</h3>
        <div className={`text-xl font-bold ${getSentimentColor(data.sentiment_score)}`}>
          {(data.sentiment_score * 100).toFixed(1)}%
        </div>
      </div>

      <div>
        <h3 className="font-bold mb-2">キーフレーズ</h3>
        <div className="flex flex-wrap gap-2">
          {data.key_phrases.map((phrase, index) => (
            <span
              key={index}
              className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm"
            >
              {phrase}
            </span>
          ))}
        </div>
      </div>

      <div>
        <h3 className="font-bold mb-2">検出されたトピック</h3>
        <div className="space-y-2">
          {data.topics.map((topic, index) => (
            <div key={index} className="flex items-center justify-between">
              <span>{topic.topic}</span>
              <span className="text-gray-600">
                {(topic.confidence * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};