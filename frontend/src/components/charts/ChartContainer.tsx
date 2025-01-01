import { ReactNode } from 'react';
import { ResponsiveContainer } from 'recharts';

interface ChartContainerProps {
  children: ReactNode;
  height?: number;
}

export function ChartContainer({ children, height = 400 }: ChartContainerProps) {
  return (
    <div className="h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        {children}
      </ResponsiveContainer>
    </div>
  );
}