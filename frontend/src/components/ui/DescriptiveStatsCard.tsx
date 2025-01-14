import React from 'react';

interface DescriptiveStats {
  mean: {
    pain: number;
    stress: number;
    sleep: number;
  };
  median: {
    pain: number;
    stress: number;
    sleep: number;
  };
  std: {
    pain: number;
    stress: number;
    sleep: number;
  };
}

interface DescriptiveStatsCardProps {
  stats: DescriptiveStats;
}

export const DescriptiveStatsCard: React.FC<DescriptiveStatsCardProps> = ({ stats }) => {
  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <div className="grid grid-cols-4 gap-4">
        <div className="col-span-1"></div>
        <div className="font-bold text-center">痛み</div>
        <div className="font-bold text-center">ストレス</div>
        <div className="font-bold text-center">睡眠</div>

        <div className="font-bold">平均</div>
        <div className="text-center">{stats.mean.pain.toFixed(2)}</div>
        <div className="text-center">{stats.mean.stress.toFixed(2)}</div>
        <div className="text-center">{stats.mean.sleep.toFixed(2)}</div>

        <div className="font-bold">中央値</div>
        <div className="text-center">{stats.median.pain.toFixed(2)}</div>
        <div className="text-center">{stats.median.stress.toFixed(2)}</div>
        <div className="text-center">{stats.median.sleep.toFixed(2)}</div>

        <div className="font-bold">標準偏差</div>
        <div className="text-center">{stats.std.pain.toFixed(2)}</div>
        <div className="text-center">{stats.std.stress.toFixed(2)}</div>
        <div className="text-center">{stats.std.sleep.toFixed(2)}</div>
      </div>
    </div>
  );
};