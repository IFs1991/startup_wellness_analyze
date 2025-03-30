import React from 'react';
import HeatMap from 'react-heatmap-grid';

interface HeatmapChartProps {
  data: number[][];
  xLabels: string[];
  yLabels: string[];
  cellSize?: number;
  cellStyle?: React.CSSProperties;
  cellRender?: (value: number) => React.ReactNode;
  xLabelWidth?: number;
}

export const HeatmapChart: React.FC<HeatmapChartProps> = ({
  data,
  xLabels,
  yLabels,
  cellSize = 30,
  cellStyle,
  cellRender,
  xLabelWidth = 60
}) => {
  return (
    <div style={{ fontSize: '13px' }}>
      <HeatMap
        xLabels={xLabels}
        yLabels={yLabels}
        data={data}
        xLabelWidth={xLabelWidth}
        cellStyle={(background, value, min, max) => ({
          background,
          fontSize: '11px',
          color: value > (max - min) / 2 ? '#fff' : '#000',
          ...cellStyle
        })}
        cellRender={cellRender}
        title={(value) => `Value: ${value}`}
        height={cellSize}
      />
    </div>
  );
};