import { useState } from 'react';
import { Input } from '@/components/ui/input';
import { Search } from 'lucide-react';
import { CompanyCard } from '@/components/companies/CompanyCard';
import { CompanyFilters } from '@/components/companies/CompanyFilters';
import type { Company } from '@/types/company';

const mockCompanies: Company[] = [
  {
    id: '1',
    name: 'テックスタート株式会社',
    industry: 'SaaS',
    stage: 'シリーズA',
    score: 85,
    employees: 45,
    location: '東京',
    foundedYear: 2020
  },
  {
    id: '2',
    name: 'ヘルスケアイノベーション',
    industry: 'ヘルスケア',
    stage: 'シリーズB',
    score: 92,
    employees: 120,
    location: '大阪',
    foundedYear: 2018
  },
  {
    id: '3',
    name: 'グリーンテック',
    industry: 'クリーンテック',
    stage: 'シード',
    score: 78,
    employees: 15,
    location: '福岡',
    foundedYear: 2022
  },
];

export function CompaniesPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [industry, setIndustry] = useState('すべて');
  const [stage, setStage] = useState('すべて');
  const [scoreRange, setScoreRange] = useState('all');

  const resetFilters = () => {
    setIndustry('すべて');
    setStage('すべて');
    setScoreRange('all');
  };

  const handleCompanyClick = (company: Company) => {
    // TODO: Implement company detail view
    console.log('Company clicked:', company);
  };

  const filteredCompanies = mockCompanies.filter(company => {
    const matchesSearch = company.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         company.industry.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesIndustry = industry === 'すべて' || company.industry === industry;
    const matchesStage = stage === 'すべて' || company.stage === stage;

    let matchesScore = true;
    if (scoreRange !== 'all') {
      const [min, max] = scoreRange.split('-').map(Number);
      matchesScore = company.score >= min && (max ? company.score <= max : true);
    }

    return matchesSearch && matchesIndustry && matchesStage && matchesScore;
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[#212121]">企業一覧</h1>
        <p className="text-muted-foreground mt-1">
          {mockCompanies.length}社の企業データを分析中
        </p>
      </div>

      <div className="space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            className="pl-10"
            placeholder="企業名、業界で検索..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <CompanyFilters
          industry={industry}
          setIndustry={setIndustry}
          stage={stage}
          setStage={setStage}
          scoreRange={scoreRange}
          setScoreRange={setScoreRange}
          onReset={resetFilters}
        />
      </div>

      <div className="grid gap-4">
        {filteredCompanies.map((company) => (
          <CompanyCard
            key={company.id}
            company={company}
            onClick={() => handleCompanyClick(company)}
          />
        ))}
      </div>
    </div>
  );
}