import { Card } from '@/components/ui/card';
import { formatCurrency } from '@/lib/utils';

interface Investment {
  date: string;
  amount: number;
  round: string;
  investors: string[];
}

interface InvestmentSummaryProps {
  investments: Investment[];
}

export function InvestmentSummary({ investments }: InvestmentSummaryProps) {
  const totalInvestment = investments.reduce((sum, inv) => sum + inv.amount, 0);

  return (
    <Card className="p-6">
      <div className="flex justify-between items-start mb-6">
        <h3 className="text-lg font-semibold">調達サマリー</h3>
        <div className="text-right">
          <p className="text-sm text-muted-foreground">調達総額</p>
          <p className="text-2xl font-bold text-primary">
            {formatCurrency(totalInvestment)}
          </p>
        </div>
      </div>

      <div className="space-y-4">
        {investments.map((investment, index) => (
          <div
            key={index}
            className="flex justify-between items-start p-4 bg-muted rounded-lg"
          >
            <div>
              <p className="font-medium">{investment.round}</p>
              <p className="text-sm text-muted-foreground">
                {new Date(investment.date).toLocaleDateString()}
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                {investment.investors.join(', ')}
              </p>
            </div>
            <p className="font-medium">{formatCurrency(investment.amount)}</p>
          </div>
        ))}
      </div>
    </Card>
  );
}