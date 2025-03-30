import React from 'react';
import { Link } from 'react-router-dom';
import CompanyCard from './CompanyCard';

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

interface CompanyListProps {
  companies: Company[];
}

const CompanyList: React.FC<CompanyListProps> = ({ companies }) => {
  if (companies.length === 0) {
    return (
      <div className="p-8 text-center bg-gray-800 rounded-lg border border-gray-700">
        <p className="text-gray-400">企業データが見つかりませんでした。検索条件を変更するか、新しい企業を追加してください。</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {companies.map(company => (
        <Link to={`/companies/${company.id}`} key={company.id} className="block">
          <CompanyCard company={company} />
        </Link>
      ))}
    </div>
  );
};

export default CompanyList;