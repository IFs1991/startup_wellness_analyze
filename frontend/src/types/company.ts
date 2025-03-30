export interface Company {
  id: string;
  name: string;
  industry: string;
  stage: string;
  score: number;
  employees: number;
  location?: string;
  foundedYear?: number;
  totalFunding?: string;
  wellnessScore?: number;
  growthRate?: string | number;
  strengths?: string[];
  weaknesses?: string[];
  scoreBreakdown?: Array<{
    category: string;
    value: number;
    description?: string;
  }>;
  revenue?: string | number;
  ceo?: string;
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

// 編集履歴のエントリを表す型
export interface EditHistoryEntry {
  id: string;
  date: string;
  field: string;
  oldValue: string;
  newValue: string;
  editedBy?: string;
}

// ステージ情報を表す型
export interface StageInfo {
  value: string;
  label: string;
  color?: string;
  description?: string;
}

export interface CompanyDetail extends Company {
  description: string;
  investments: Investment[];
  fundingRounds?: number;
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
  // 編集履歴を追加
  editHistory?: EditHistoryEntry[];
}