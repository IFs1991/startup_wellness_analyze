import React from 'react';
import { ChevronRight, TrendingUp } from 'lucide-react';

interface Company {
  id: string;
  name: string;
  industry: string;
  stage: string;
  location: string;
  employeesCount: number;
  wellnessScore: number;
  growthRate: number;
}

interface CompanyCardProps {
  company: Company;
}

const CompanyCard: React.FC<CompanyCardProps> = ({ company }) => {
  // スコアに基づいた色を取得する関数
  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-500';
    if (score >= 80) return 'text-blue-500';
    if (score >= 70) return 'text-amber-500';
    return 'text-red-500';
  };

  // 成長率の色を取得する関数
  const getGrowthColor = (rate: number) => {
    if (rate >= 20) return 'text-green-500';
    if (rate >= 10) return 'text-blue-500';
    if (rate >= 5) return 'text-amber-500';
    return 'text-gray-400';
  };

  return (
    <div className="border border-gray-800 rounded-lg bg-gray-800 hover:bg-gray-750 transition-all p-4 flex justify-between items-center">
      <div className="flex-1">
        <div className="flex items-center">
          <h2 className="text-lg font-semibold">{company.name}</h2>
          <span className="ml-3 px-2 py-1 text-xs rounded bg-gray-700 text-white">
            {company.stage}
          </span>
        </div>
        <div className="text-sm text-gray-400 mt-1">
          {company.industry} • {company.employeesCount}名 • {company.location}
        </div>
      </div>
      <div className="flex items-center">
        <div className="mr-4 text-right">
          <div className="flex items-center justify-end mb-1">
            <TrendingUp className={`h-4 w-4 mr-1 ${getGrowthColor(company.growthRate)}`} />
            <span className={`text-sm font-medium ${getGrowthColor(company.growthRate)}`}>
              {company.growthRate}%
            </span>
          </div>
          <div className="text-xs text-gray-400">ウェルネススコア</div>
          <div className={`text-2xl font-bold ${getScoreColor(company.wellnessScore)}`}>
            {company.wellnessScore}
          </div>
        </div>
        <ChevronRight className="h-5 w-5 text-gray-500" />
      </div>
    </div>
  );
};

export default CompanyCard;