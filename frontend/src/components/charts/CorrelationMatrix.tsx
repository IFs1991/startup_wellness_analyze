import { useMemo } from 'react';
import {
  Card,
  CardContent
} from '@/components/ui/card';

interface CorrelationData {
  x: string;
  y: string;
  value: number;
}

interface CorrelationMatrixProps {
  data: CorrelationData[];
}

export function CorrelationMatrix({ data }: CorrelationMatrixProps) {
  // X軸とY軸の一意の値を取得
  const { xLabels, yLabels } = useMemo(() => {
    const xLabels = Array.from(new Set(data.map(d => d.x)));
    const yLabels = Array.from(new Set(data.map(d => d.y)));
    return { xLabels, yLabels };
  }, [data]);

  // 値からカラーコードを生成する関数
  const getColorFromValue = (value: number) => {
    // 相関が強いほど青の濃さが増す
    const intensity = Math.abs(value);
    const r = value < 0 ? Math.round(255 * intensity) : 0;
    const g = 0;
    const b = value > 0 ? Math.round(255 * intensity) : 0;
    return `rgba(${r}, ${g}, ${b}, ${intensity})`;
  };

  // セルのテキスト色を決定する（背景色に基づいて）
  const getTextColor = (value: number) => {
    return Math.abs(value) > 0.5 ? 'text-white' : 'text-gray-700';
  };

  return (
    <Card>
      <CardContent className="p-4">
        <div className="text-sm font-medium mb-2">相関係数行列</div>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="p-2 border"></th>
                {yLabels.map(label => (
                  <th key={label} className="p-2 border text-sm">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {xLabels.map(x => (
                <tr key={x}>
                  <td className="p-2 border font-medium text-sm">{x}</td>
                  {yLabels.map(y => {
                    const cell = data.find(d => d.x === x && d.y === y);
                    const value = cell ? cell.value : 0;
                    const bgColor = getColorFromValue(value);
                    const textColor = getTextColor(value);

                    return (
                      <td
                        key={`${x}-${y}`}
                        className={`p-2 border text-center text-sm ${textColor}`}
                        style={{ backgroundColor: bgColor }}
                      >
                        {value.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex justify-between items-center mt-3 text-xs text-gray-500">
          <div>負の相関</div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-red-600 mr-1"></div>
            <div className="w-4 h-4 bg-red-300 mr-1"></div>
            <div className="w-4 h-4 bg-gray-100 mr-1"></div>
            <div className="w-4 h-4 bg-blue-300 mr-1"></div>
            <div className="w-4 h-4 bg-blue-600 mr-1"></div>
          </div>
          <div>正の相関</div>
        </div>
      </CardContent>
    </Card>
  );
}