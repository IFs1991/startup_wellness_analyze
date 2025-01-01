interface DataPoint {
  name: string;
  value: number;
}

interface DataTableProps {
  data: DataPoint[];
}

export function DataTable({ data }: DataTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr>
            <th className="text-left p-2 text-muted-foreground font-medium">期間</th>
            <th className="text-left p-2 text-muted-foreground font-medium">値</th>
          </tr>
        </thead>
        <tbody>
          {data.map((item) => (
            <tr key={item.name} className="border-b border-border last:border-0">
              <td className="p-2">{item.name}</td>
              <td className="p-2">{item.value.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}