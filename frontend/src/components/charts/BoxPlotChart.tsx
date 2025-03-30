import React from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Rectangle,
  RectangleProps
} from 'recharts';

interface BoxPlotData {
  name: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  [key: string]: any;
}

interface BoxPlotChartProps {
  data: BoxPlotData[];
  width?: number;
  height?: number;
  boxWidth?: number;
}

// Custom BoxPlot shape component
const BoxPlotShape = (props: RectangleProps & {
  dataKey?: string;
  Q1?: number;
  median?: number;
  Q3?: number;
  min?: number;
  max?: number;
}) => {
  const { x, y, width, height, Q1, median, Q3, min, max } = props;

  if (!x || !y || !width || !height ||
      Q1 === undefined || median === undefined ||
      Q3 === undefined || min === undefined || max === undefined) {
    return null;
  }

  const centerX = (x || 0) + (width || 0) / 2;

  return (
    <g>
      {/* Min to Max vertical line */}
      <line
        x1={centerX}
        y1={y - height + min * height}
        x2={centerX}
        y2={y - height + max * height}
        stroke="#000"
        strokeWidth={1}
      />

      {/* Box from Q1 to Q3 */}
      <rect
        x={x}
        y={y - height + Q1 * height}
        width={width}
        height={(Q3 - Q1) * height}
        fill="#8884d8"
        stroke="#000"
        strokeWidth={1}
        opacity={0.8}
      />

      {/* Median line */}
      <line
        x1={x}
        y1={y - height + median * height}
        x2={x + width}
        y2={y - height + median * height}
        stroke="#000"
        strokeWidth={1.5}
      />

      {/* Min horizontal line */}
      <line
        x1={centerX - width / 4}
        y1={y - height + min * height}
        x2={centerX + width / 4}
        y2={y - height + min * height}
        stroke="#000"
        strokeWidth={1}
      />

      {/* Max horizontal line */}
      <line
        x1={centerX - width / 4}
        y1={y - height + max * height}
        x2={centerX + width / 4}
        y2={y - height + max * height}
        stroke="#000"
        strokeWidth={1}
      />
    </g>
  );
};

export const BoxPlotChart: React.FC<BoxPlotChartProps> = ({
  data,
  width = 800,
  height = 400,
  boxWidth = 30
}) => {
  const normalizedData = data.map(item => {
    const range = item.max - item.min;
    return {
      ...item,
      _min: 0,
      _q1: (item.q1 - item.min) / range,
      _median: (item.median - item.min) / range,
      _q3: (item.q3 - item.min) / range,
      _max: 1
    };
  });

  return (
    <ComposedChart
      width={width}
      height={height}
      data={normalizedData}
      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
    >
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis domain={[0, 'dataMax']} />
      <Tooltip
        formatter={(value, name, props) => {
          const item = data[props.index];
          if (name === 'boxPlot') return '';

          return [
            `Min: ${item.min}\nQ1: ${item.q1}\nMedian: ${item.median}\nQ3: ${item.q3}\nMax: ${item.max}`,
            item.name
          ];
        }}
      />
      <Legend />
      <Bar
        dataKey="_q1"
        name="boxPlot"
        shape={(props) => (
          <BoxPlotShape
            {...props}
            Q1={normalizedData[props.index]._q1}
            median={normalizedData[props.index]._median}
            Q3={normalizedData[props.index]._q3}
            min={normalizedData[props.index]._min}
            max={normalizedData[props.index]._max}
          />
        )}
        barSize={boxWidth}
      />
    </ComposedChart>
  );
};