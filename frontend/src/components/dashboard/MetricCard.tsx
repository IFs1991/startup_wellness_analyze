import { Card } from '@/components/ui/card';
import { ArrowUpIcon, ArrowDownIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  trend?: 'up' | 'down';
  description?: string;
}

export function MetricCard({ 
  title, 
  value, 
  change, 
  trend,
  description 
}: MetricCardProps) {
  return (
    <Card className="p-6">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
          <div className="mt-2 flex items-baseline">
            <p className="text-2xl font-semibold">{value}</p>
            {change && (
              <span className={cn(
                "ml-2 text-sm font-medium",
                trend === 'up' ? 'text-green-600' : 'text-red-600'
              )}>
                {trend === 'up' ? <ArrowUpIcon className="h-4 w-4 inline" /> : <ArrowDownIcon className="h-4 w-4 inline" />}
                {change}%
              </span>
            )}
          </div>
        </div>
      </div>
      {description && (
        <p className="mt-2 text-sm text-muted-foreground">{description}</p>
      )}
    </Card>
  );
}