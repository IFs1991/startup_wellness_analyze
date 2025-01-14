import { Card } from '@/components/ui/card';
import { CalendarDays, Users, MapPin, TrendingUp } from 'lucide-react';

interface CompanyOverviewProps {
  foundedYear: number;
  employees: number;
  location: string;
  growth: number;
}

export function CompanyOverview({ foundedYear, employees, location, growth }: CompanyOverviewProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card className="p-4">
        <div className="flex items-center space-x-3">
          <CalendarDays className="h-5 w-5 text-muted-foreground" />
          <div>
            <p className="text-sm font-medium text-muted-foreground">設立年</p>
            <p className="text-lg font-semibold">{foundedYear}年</p>
          </div>
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center space-x-3">
          <Users className="h-5 w-5 text-muted-foreground" />
          <div>
            <p className="text-sm font-medium text-muted-foreground">従業員数</p>
            <p className="text-lg font-semibold">{employees}名</p>
          </div>
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center space-x-3">
          <MapPin className="h-5 w-5 text-muted-foreground" />
          <div>
            <p className="text-sm font-medium text-muted-foreground">所在地</p>
            <p className="text-lg font-semibold">{location}</p>
          </div>
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center space-x-3">
          <TrendingUp className="h-5 w-5 text-muted-foreground" />
          <div>
            <p className="text-sm font-medium text-muted-foreground">成長率</p>
            <p className="text-lg font-semibold text-green-600">+{growth}%</p>
          </div>
        </div>
      </Card>
    </div>
  );
}