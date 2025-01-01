import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface DashboardCardProps {
  title: string;
  description: string;
  buttonText: string;
  onClick?: () => void;
}

export function DashboardCard({ title, description, buttonText, onClick }: DashboardCardProps) {
  return (
    <Card className="p-6 shadow-sm hover:shadow-md transition-shadow">
      <h2 className="text-[#212121] text-xl font-semibold mb-4">{title}</h2>
      <p className="text-[#757575] mb-4">{description}</p>
      <Button 
        className="w-full bg-[#4285F4] hover:bg-[#3367D6] text-white"
        onClick={onClick}
      >
        {buttonText}
      </Button>
    </Card>
  );
}