import { useNavigate } from 'react-router-dom';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ChevronRight } from 'lucide-react';
import { getScoreColor } from '@/lib/utils';
import type { Company } from '@/types/company';

interface CompanyCardProps {
  company: Company;
  onClick: () => void;
}

export function CompanyCard({ company, onClick }: CompanyCardProps) {
  const navigate = useNavigate();
  const scoreColor = getScoreColor(company.score);

  return (
    <Card
      className="p-4 hover:shadow-md transition-shadow cursor-pointer"
      onClick={onClick}
    >
      <div className="flex justify-between items-start">
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <h3 className="font-semibold text-lg">{company.name}</h3>
            <Badge variant="outline">{company.stage}</Badge>
          </div>

          <div className="flex space-x-2 text-sm text-muted-foreground">
            <span>{company.industry}</span>
            <span>•</span>
            <span>{company.employees}名</span>
            {company.location && (
              <>
                <span>•</span>
                <span>{company.location}</span>
              </>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="text-right">
            <div className="text-sm font-medium">ウェルネススコア</div>
            <div className={`text-2xl font-bold ${scoreColor}`}>
              {company.score}
            </div>
          </div>

          <Button variant="ghost" size="icon">
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </Card>
  );
}