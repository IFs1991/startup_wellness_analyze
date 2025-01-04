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

interface FinancialHistory {
  date: string;
  revenue: number;
  profit: number;
  assets: number;
  liabilities: number;
}

interface Investment {
  id: string;
  date: string;
  amount: number;
  round: string;
  investors: string[];
}

export interface CompanyDetail extends Company {
  description: string;
  investments: Investment[];
  financials: {
    revenue: number;
    growth: number;
    profit: number;
    history: FinancialHistory[];
  };
  wellness: {
    score: number;
    engagement: number;
    satisfaction: number;
    workLife: number;
    stress: number;
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