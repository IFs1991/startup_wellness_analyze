import React from 'react';

interface DescriptiveStats {
  mean: {
    pain_level: number;
    stress_level: number;
    sleep_quality: number;
  };
  median: {
    pain_level: number;
    stress_level: number;
    sleep_quality: number;
  };
  std: {
    pain_level: number;
    stress_level: number;
    sleep_quality: number;
  };
}

interface DescriptiveStatsCardProps {
  stats: DescriptiveStats;
}

export const DescriptiveStatsCard: React.FC<DescriptiveStatsCardProps> = ({ stats }) => {
  const formatNumber = (num: number) => num.toFixed(2);

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold mb-2">平均値</h3>
          <div className="grid grid-cols-2 gap-2">
            <div>痛みレベル:</div>
            <div>{formatNumber(stats.mean.pain_level)}</div>
            <div>ストレスレベル:</div>
            <div>{formatNumber(stats.mean.stress_level)}</div>
            <div>睡眠の質:</div>
            <div>{formatNumber(stats.mean.sleep_quality)}</div>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-2">中央値</h3>
          <div className="grid grid-cols-2 gap-2">
            <div>痛みレベル:</div>
            <div>{formatNumber(stats.median.pain_level)}</div>
            <div>ストレスレベル:</div>
            <div>{formatNumber(stats.median.stress_level)}</div>
            <div>睡眠の質:</div>
            <div>{formatNumber(stats.median.sleep_quality)}</div>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-2">標準偏差</h3>
          <div className="grid grid-cols-2 gap-2">
            <div>痛みレベル:</div>
            <div>{formatNumber(stats.std.pain_level)}</div>
            <div>ストレスレベル:</div>
            <div>{formatNumber(stats.std.stress_level)}</div>
            <div>睡眠の質:</div>
            <div>{formatNumber(stats.std.sleep_quality)}</div>
          </div>
        </div>
      </div>
    </div>
  );
};