import { useEffect, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';

interface Series {
  name: string;
  data: number[];
  color: string;
  isMainMetric: boolean;
  scale?: number; // 異なる単位のデータをスケーリングするための係数
}

interface Annotation {
  date: string;
  text: string;
  impact: 'positive' | 'negative' | 'neutral';
}

export interface TimeSeriesData {
  dates: string[];
  series: Series[];
  annotations: Annotation[];
  insights: string[];
}

interface TimeSeriesChartProps {
  data: TimeSeriesData;
}

function drawTimeSeriesChart(
  canvas: HTMLCanvasElement,
  data: TimeSeriesData
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // 高解像度ディスプレイ対応
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  // キャンバスをクリア
  ctx.clearRect(0, 0, rect.width, rect.height);

  // パディングの設定
  const padding = {
    top: 20,
    right: 30,
    bottom: 50,
    left: 50
  };

  // 描画領域の計算
  const chartWidth = rect.width - padding.left - padding.right;
  const chartHeight = rect.height - padding.top - padding.bottom;

  // データの最小値と最大値を計算
  let minValue = Infinity;
  let maxValue = -Infinity;

  data.series.forEach(series => {
    const scale = series.scale || 1;
    const scaledData = series.data.map(value => value * scale);
    const seriesMin = Math.min(...scaledData);
    const seriesMax = Math.max(...scaledData);

    if (seriesMin < minValue) minValue = seriesMin;
    if (seriesMax > maxValue) maxValue = seriesMax;
  });

  // 余白を追加
  const valueRange = maxValue - minValue;
  minValue = Math.max(0, minValue - valueRange * 0.1);
  maxValue = maxValue + valueRange * 0.1;

  // X軸目盛りの計算
  const dateCount = data.dates.length;
  const xStep = chartWidth / (dateCount - 1);

  // Y軸目盛りの計算
  const yAxisSteps = 5;
  const yStep = chartHeight / yAxisSteps;
  const valueStep = (maxValue - minValue) / yAxisSteps;

  // グリッド線の描画
  ctx.beginPath();
  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth = 0.5;

  // 水平グリッド線
  for (let i = 0; i <= yAxisSteps; i++) {
    const y = padding.top + chartHeight - (i * yStep);
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + chartWidth, y);
  }

  // 垂直グリッド線（日付ごと）
  for (let i = 0; i < dateCount; i++) {
    const x = padding.left + i * xStep;
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, padding.top + chartHeight);
  }

  ctx.stroke();

  // Y軸ラベルの描画
  ctx.fillStyle = '#6b7280';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  for (let i = 0; i <= yAxisSteps; i++) {
    const value = minValue + i * valueStep;
    const y = padding.top + chartHeight - (i * yStep);
    ctx.fillText(value.toFixed(1), padding.left - 10, y);
  }

  // X軸ラベルの描画（日付）
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';

  // 日付の表示間隔を調整（データポイントが多い場合）
  const labelInterval = dateCount > 12 ? 2 : 1;

  for (let i = 0; i < dateCount; i += labelInterval) {
    const x = padding.left + i * xStep;
    const date = data.dates[i];
    ctx.fillText(date, x, padding.top + chartHeight + 10);
  }

  // 注釈マーカーの描画
  data.annotations.forEach(annotation => {
    const dateIndex = data.dates.indexOf(annotation.date);
    if (dateIndex !== -1) {
      const x = padding.left + dateIndex * xStep;

      // マーカー
      ctx.beginPath();
      ctx.arc(x, padding.top, 6, 0, 2 * Math.PI);
      ctx.fillStyle = annotation.impact === 'positive' ? 'rgba(34, 197, 94, 0.7)' : 'rgba(239, 68, 68, 0.7)';
      ctx.fill();

      // マーカーの線
      ctx.beginPath();
      ctx.strokeStyle = annotation.impact === 'positive' ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartHeight);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  });

  // 各系列のデータプロット
  data.series.forEach(series => {
    const scale = series.scale || 1;

    // ライン描画
    ctx.beginPath();
    ctx.strokeStyle = series.color;
    ctx.lineWidth = series.isMainMetric ? 3 : 2;

    // 最初のポイント
    const x0 = padding.left;
    const yValue0 = (series.data[0] * scale - minValue) / (maxValue - minValue) * chartHeight;
    const y0 = padding.top + chartHeight - yValue0;
    ctx.moveTo(x0, y0);

    // 残りのポイント
    for (let i = 1; i < series.data.length; i++) {
      const x = padding.left + i * xStep;
      const yValue = (series.data[i] * scale - minValue) / (maxValue - minValue) * chartHeight;
      const y = padding.top + chartHeight - yValue;
      ctx.lineTo(x, y);
    }

    ctx.stroke();

    // データポイントのプロット
    if (series.isMainMetric) {
      for (let i = 0; i < series.data.length; i++) {
        const x = padding.left + i * xStep;
        const yValue = (series.data[i] * scale - minValue) / (maxValue - minValue) * chartHeight;
        const y = padding.top + chartHeight - yValue;

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = series.color;
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  });

  // 凡例の描画
  const legendY = padding.top + chartHeight + 30;
  const legendItemWidth = chartWidth / data.series.length;

  data.series.forEach((series, index) => {
    const x = padding.left + (index * legendItemWidth) + 15;

    // 凡例の色マーカー
    ctx.beginPath();
    ctx.rect(x, legendY, 10, 10);
    ctx.fillStyle = series.color;
    ctx.fill();

    // 凡例のテキスト
    ctx.fillStyle = '#374151';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(series.name, x + 15, legendY + 5);
  });
}

export function TimeSeriesChart({ data }: TimeSeriesChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current) {
      drawTimeSeriesChart(canvasRef.current, data);
    }

    const handleResize = () => {
      if (canvasRef.current) {
        drawTimeSeriesChart(canvasRef.current, data);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [data]);

  return (
    <Card>
      <CardContent className="p-4">
        <canvas
          ref={canvasRef}
          style={{ width: '100%', height: '300px' }}
        />
      </CardContent>
    </Card>
  );
}