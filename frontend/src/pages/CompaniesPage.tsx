import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { PlusCircle, Filter, Search } from 'lucide-react';
import CompanyList from '../components/companies/CompanyList';
import CompanyFilters from '../components/companies/CompanyFilters';
import AddCompanyDialog from '../components/companies/AddCompanyDialog';

// モック企業データの型定義
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

const CompaniesPage: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [selectedFilters, setSelectedFilters] = useState({
    industry: '',
    stage: '',
    location: '',
  });

  // モックデータ
  const companies: Company[] = [
    {
      id: '1',
      name: 'テックスタート株式会社',
      industry: 'SaaS',
      employeesCount: 45,
      location: '東京',
      stage: 'シリーズA',
      wellnessScore: 85,
      growthRate: 15
    },
    {
      id: '2',
      name: 'ヘルスケアイノベーション',
      industry: 'ヘルスケア',
      employeesCount: 120,
      location: '大阪',
      stage: 'シリーズB',
      wellnessScore: 92,
      growthRate: 25
    },
    {
      id: '3',
      name: 'グリーンテック',
      industry: 'クリーンテック',
      employeesCount: 15,
      location: '福岡',
      stage: 'シード',
      wellnessScore: 78,
      growthRate: 8
    },
    {
      id: '4',
      name: 'フューチャーデザイン',
      industry: 'デザイン',
      employeesCount: 28,
      location: '京都',
      stage: 'シリーズA',
      wellnessScore: 81,
      growthRate: 12
    },
    {
      id: '5',
      name: 'ソフトイノベーションズ',
      industry: 'ソフトウェア',
      employeesCount: 52,
      location: '名古屋',
      stage: 'シリーズB',
      wellnessScore: 76,
      growthRate: 18
    },
  ];

  // フィルタリングされた企業リスト
  const filteredCompanies = companies.filter((company) => {
    const matchesSearch = company.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          company.industry.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesIndustry = selectedFilters.industry === '' || company.industry === selectedFilters.industry;
    const matchesStage = selectedFilters.stage === '' || company.stage === selectedFilters.stage;
    const matchesLocation = selectedFilters.location === '' || company.location === selectedFilters.location;

    return matchesSearch && matchesIndustry && matchesStage && matchesLocation;
  });

  // フィルターオプション
  const filterOptions = {
    industries: [...new Set(companies.map(c => c.industry))],
    stages: [...new Set(companies.map(c => c.stage))],
    locations: [...new Set(companies.map(c => c.location))],
  };

  const handleFilterChange = (name: string, value: string) => {
    setSelectedFilters(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const toggleFilters = () => {
    setShowFilters(!showFilters);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* ヘッダーセクション */}
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold mb-2">企業一覧</h1>
            <p className="text-gray-400">{filteredCompanies.length}社の企業データを分析中</p>
          </div>
          <button
            onClick={() => setAddDialogOpen(true)}
            className="flex items-center bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition duration-150"
          >
            <PlusCircle className="mr-2 h-5 w-5" />
            企業を追加
          </button>
        </div>

        {/* 検索フィルターセクション */}
        <div className="mb-6">
          <div className="relative mb-4">
            <input
              type="text"
              placeholder="企業名、業界で検索..."
              className="w-full bg-gray-800 border border-gray-700 rounded-lg py-2 px-4 pl-10 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
            <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-500" />
            <button
              onClick={toggleFilters}
              className="absolute right-3 top-2.5 flex items-center text-gray-400 hover:text-white"
            >
              <Filter className="h-5 w-5 mr-1" />
              フィルター
            </button>
          </div>

          {showFilters && (
            <CompanyFilters
              options={filterOptions}
              selected={selectedFilters}
              onChange={handleFilterChange}
            />
          )}
        </div>

        {/* 企業リスト */}
        <CompanyList companies={filteredCompanies} />

        {/* 企業追加ダイアログ */}
        {addDialogOpen && (
          <AddCompanyDialog
            open={addDialogOpen}
            onClose={() => setAddDialogOpen(false)}
            onAddCompany={(company) => {
              console.log('New company:', company);
              setAddDialogOpen(false);
            }}
          />
        )}
      </div>
    </div>
  );
};

export default CompaniesPage;