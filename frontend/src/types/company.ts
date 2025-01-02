export interface Company {
  id: string;
  name: string;
  industry: string;
  stage: string;
  score: number;
  employees: number;
  location?: string;
  foundedYear?: number;
}

export interface CompanyDetail extends Company {
  description: string;
  financials: {
    revenue: number;
    growth: number;
    profit: number;
  };
  wellness: {
    score: number;
    engagement: number;
    satisfaction: number;
    trends: Array<{
      date: string;
      score: number;
    }>;
  };
  surveys: Array<{
    id: string;
    date: string;
    responseRate: number;
    averageScore: number;
  }>;
}