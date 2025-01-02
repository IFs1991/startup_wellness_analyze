import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { companiesApi } from '@/lib/api/companies';
import { CompanyOverview } from '@/components/companies/CompanyDetail/CompanyOverview';
import { FinancialMetrics } from '@/components/companies/CompanyDetail/FinancialMetrics';
import { WellnessAnalysis } from '@/components/companies/CompanyDetail/WellnessAnalysis';
import { FinancialDataUpload } from '@/components/companies/CompanyDetail/FinancialDataUpload';
import { InvestmentSummary } from '@/components/companies/CompanyDetail/InvestmentSummary';
import { CompanyNotes } from '@/components/companies/CompanyDetail/CompanyNotes';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Pencil } from 'lucide-react';
import type { CompanyDetail } from '@/types/company';

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
        foundedYear={company.foundedYear}
        employees={company.employees}
        location={company.location}
        growth={company.financials.growth}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <InvestmentSummary investments={company.investments} />
        <CompanyNotes companyId={company.id} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FinancialMetrics data={company.financials.history} />
        <WellnessAnalysis scores={company.wellness} />
      </div>

      <FinancialDataUpload
        companyId={company.id}
        onUploadComplete={handleUploadComplete}
      />
    </div>
  );
}

// ... (既存のCompanyDetailSkeletonコンポーネント)