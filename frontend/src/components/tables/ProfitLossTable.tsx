import React from 'react';

interface ProfitLossData {
  date: string;
  revenue: number;
  expenses: number;
  profit: number;
}

interface ProfitLossTableProps {
  data: ProfitLossData[];
}

export const ProfitLossTable: React.FC<ProfitLossTableProps> = ({ data }) => {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full bg-white border border-gray-300">
        <thead>
          <tr className="bg-gray-100">
            <th className="px-4 py-2 border">日付</th>
            <th className="px-4 py-2 border">収益</th>
            <th className="px-4 py-2 border">支出</th>
            <th className="px-4 py-2 border">利益</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, index) => (
            <tr key={index} className="hover:bg-gray-50">
              <td className="px-4 py-2 border">{new Date(row.date).toLocaleDateString('ja-JP')}</td>
              <td className="px-4 py-2 border text-right">¥{row.revenue.toLocaleString()}</td>
              <td className="px-4 py-2 border text-right">¥{row.expenses.toLocaleString()}</td>
              <td className="px-4 py-2 border text-right">¥{row.profit.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};