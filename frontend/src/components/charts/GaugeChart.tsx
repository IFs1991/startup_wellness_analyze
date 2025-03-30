import React from 'react';
import GaugeComponent from 'react-gauge-component';

interface GaugeChartProps {
  value: number;
  min?: number;
  max?: number;
  label?: string;
  size?: string;
}

export const GaugeChart: React.FC<GaugeChartProps> = ({
  value,
  min = 0,
  max = 100,
  label = '',
  size = '175px'
}) => {
  // Calculate percentage for the gauge
  const percentage = Math.min(Math.max((value - min) / (max - min), 0), 1);

  return (
    <div style={{ width: size, height: size }}>
      <GaugeComponent
        value={percentage * 100}
        type="radial"
        labels={{
          markLabel: {
            type: 'outer',
            marks: [
              { value: 0, label: `${min}` },
              { value: 25 },
              { value: 50 },
              { value: 75 },
              { value: 100, label: `${max}` }
            ]
          },
          valueLabel: {
            formatTextValue: () => `${value}`
          }
        }}
        arc={{
          colorArray: ['#5BE12C', '#F5CD19', '#EA4228'],
          subArcs: [{ limit: 40 }, { limit: 70 }, { limit: 100 }]
        }}
        pointer={{ elastic: true }}
      />
      {label && <div className="text-center mt-2">{label}</div>}
    </div>
  );
};