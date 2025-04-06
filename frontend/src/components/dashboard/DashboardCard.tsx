import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface DashboardCardProps {
  title: string;
  subtitle?: string;
  actionText?: string;
  actionIcon?: React.ReactNode;
  children?: React.ReactNode;
}

export function DashboardCard({
  title,
  subtitle,
  actionText,
  actionIcon,
  children
}: DashboardCardProps) {
  return (
    <Card className="p-6 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-[#212121] text-xl font-semibold">{title}</h2>
          {subtitle && <p className="text-[#757575] mt-1">{subtitle}</p>}
        </div>
        {actionText && actionIcon && (
          <Button
            variant="ghost"
            className="flex items-center gap-2"
          >
            {actionText}
            {actionIcon}
          </Button>
        )}
      </div>
      {children}
    </Card>
  );
}