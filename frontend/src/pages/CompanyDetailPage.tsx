import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { companiesApi } from '@/lib/api/companies';
import { CompanyOverview } from '@/components/companies/CompanyDetail/CompanyOverview';
import { FinancialMetrics } from '@/components/companies/CompanyDetail/FinancialMetrics';
import { WellnessAnalysis } from '@/components/companies/CompanyDetail/WellnessAnalysis';
import { FinancialDataUpload } from '@/components/companies/CompanyDetail/FinancialDataUpload';
import { InvestmentSummary } from '@/components/companies/CompanyDetail/InvestmentSummary';
import { CompanyNotes } from '@/components/companies/CompanyDetail/CompanyNotes';
import { CompanyDetailSkeleton } from '@/components/companies/CompanyDetail/CompanyDetailSkeleton';
import { AIChat } from '@/components/companies/CompanyDetail/AIChat';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Pencil } from 'lucide-react';
import type { CompanyDetail } from '@/types/company';

interface WellnessScore {
  score: number;
  engagement: number;
  satisfaction: number;
  workLife: number;
  stress: number;
  overall: number;
  trends: Array<{
    date: string;
    score: number;
  }>;
}

export function CompanyDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [company, setCompany] = useState<CompanyDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchCompanyDetail() {
      if (!id) return;
      try {
        const data = await companiesApi.getCompanyById(id);
        setCompany(data);
      } catch (error) {
        console.error('Failed to fetch company details:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchCompanyDetail();
  }, [id]);

  const handleEditClick = () => {
    // TODO: Implement edit functionality
  };

  const handleUploadComplete = () => {
    // Refresh financial data
  };

  if (loading) {
    return <CompanyDetailSkeleton />;
  }

  if (!company) {
    return <div>企業情報が見つかりませんでした。</div>;
  }

  const wellnessScores: WellnessScore = {
    ...company.wellness,
    overall: company.wellness.score
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-2xl font-bold">{company.name}</h1>
            <Badge variant="outline">{company.stage}</Badge>
          </div>
          <p className="text-muted-foreground mt-2">{company.description}</p>
        </div>
        <Button onClick={handleEditClick} variant="outline">
          <Pencil className="h-4 w-4 mr-2" />
          編集
        </Button>
      </div>

      <CompanyOverview
        foundedYear={company.foundedYear || 0}
        employees={company.employees}
        location={company.location || '未設定'}
        growth={company.financials.growth}
      />

      <div className="grid grid-cols-2 gap-6">
        <div className="space-y-6">
          <FinancialMetrics data={company.financials.history} />
          <InvestmentSummary investments={company.investments || []} />
        </div>
        <div className="space-y-6">
          <WellnessAnalysis scores={wellnessScores} />
          <AIChat companyId={company.id} />
        </div>
      </div>

      <FinancialDataUpload
        companyId={company.id}
        onUploadComplete={handleUploadComplete}
      />
    </div>
  );
}