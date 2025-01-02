import type { CompanyDetail } from '@/types/company';

export const mockCompanyDetails: Record<string, CompanyDetail> = {
  '1': {
    id: '1',
    name: 'テックスタート株式会社',
    industry: 'SaaS',
    stage: 'シリーズA',
    score: 85,
    employees: 45,
    location: '東京',
    foundedYear: 2020,
    description: 'クラウドベースの業務効率化ソリューションを提供するSaaS企業',
    financials: {
      revenue: 250000000,
      growth: 156,
      profit: 50000000,
      history: [
        { date: '2023-Q1', revenue: 150000000, profit: 30000000, assets: 200000000, liabilities: 80000000 },
        { date: '2023-Q2', revenue: 180000000, profit: 36000000, assets: 220000000, liabilities: 85000000 },
        { date: '2023-Q3', revenue: 220000000, profit: 44000000, assets: 250000000, liabilities: 90000000 },
        { date: '2023-Q4', revenue: 250000000, profit: 50000000, assets: 280000000, liabilities: 95000000 },
      ]
    },
    wellness: {
      score: 85,
      engagement: 92,
      satisfaction: 88,
      workLife: 83,
      stress: 78,
      trends: [
        { date: '2023-10', score: 82 },
        { date: '2023-11', score: 84 },
        { date: '2023-12', score: 85 },
        { date: '2024-01', score: 85 },
      ]
    },
    surveys: [
      {
        id: 's1',
        date: '2024-01',
        responseRate: 95,
        averageScore: 4.2
      }
    ]
  }
};