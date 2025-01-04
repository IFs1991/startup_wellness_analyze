import React from 'react';

interface CorrelationData {
  pain_stress: number;
  pain_sleep: number;
  stress_sleep: number;
}

interface CorrelationMatrixProps {
  data: CorrelationData;
}

export const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({ data }) => {
  const getCorrelationColor = (value: number) => {
    const absValue = Math.abs(value);
    const intensity = Math.min(Math.abs(value) * 255, 255);
    return value >= 0
      ? `rgb(${intensity}, ${intensity}, 255)`
      : `rgb(255, ${intensity}, ${intensity})`;
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <table className="w-full">
        <thead>
          <tr>
            <th className="px-4 py-2"></th>
            <th className="px-4 py-2">痛み</th>
            <th className="px-4 py-2">ストレス</th>
            <th className="px-4 py-2">睡眠</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="px-4 py-2 font-bold">痛み</td>
            <td className="px-4 py-2 text-center" style={{ backgroundColor: getCorrelationColor(1) }}>1.00</td>
            <td className="px-4 py-2 text-center" style={{ backgroundColor: getCorrelationColor(data.pain_stress) }}>
              {data.pain_stress.toFixed(2)}
            </td>
            <td className="px-4 py-2 text-center" style={{ backgroundColor: getCorrelationColor(data.pain_sleep) }}>
              {data.pain_sleep.toFixed(2)}
            </td>
          </tr>
          <tr>
            <td className="px-4 py-2 font-bold">ストレス</td>
            <td className="px-4 py-2 text-center" style={{ backgroundColor: getCorrelationColor(data.pain_stress) }}>
              {data.pain_stress.toFixed(2)}
            </td>
            <td className="px-4 py-2 text-center" style={{ backgroundColor: getCorrelationColor(1) }}>1.00</td>
            <td className="px-4 py-2 text-center" style={{ backgroundColor: getCorrelationColor(data.stress_sleep) }}>
              {data.stress_sleep.toFixed(2)}
            </td>
          </tr>
          <tr>
            <td className="px-4 py-2 font-bold">睡眠</td>
            <td className="px-4 py-2 text-center" style={{ backgroundColor: getCorrelationColor(data.pain_sleep) }}>
              {data.pain_sleep.toFixed(2)}
            </td>
            <td className="px-4 py-2 text-center" style={{ backgroundColor: getCorrelationColor(data.stress_sleep) }}>
              {data.stress_sleep.toFixed(2)}
            </td>
            <td className="px-4 py-2 text-center" style={{ backgroundColor: getCorrelationColor(1) }}>1.00</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};