import { useEffect, useRef, memo } from 'react';

interface Point {
  x: number;
  y: number;
}

interface RegressionLine {
  slope: number;
  intercept: number;
  r2: number;
}

interface ScatterPlotChartProps {
  data: Point[];
  regressionLine: RegressionLine;
  xLabel: string;
  yLabel: string;
  height?: number;
}

// メモ化したコンポーネント
const ScatterPlotChart = memo(function ScatterPlotChart({
  data,
  regressionLine,
  xLabel,
  yLabel,
  height = 300
}: ScatterPlotChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // キャンバスのクリア
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // パディングの設定
    const padding = { top: 20, right: 30, bottom: 50, left: 60 };

    // データの最小値と最大値を計算
    const xValues = data.map(d => d.x);
    const yValues = data.map(d => d.y);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    // X軸とY軸のスケールを設定
    const xScale = (canvas.width - padding.left - padding.right) / (xMax - xMin);
    const yScale = (canvas.height - padding.top - padding.bottom) / (yMax - yMin);

    // グリッド線を描画
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 0.5;

    // X軸のグリッド線
    const xStep = Math.ceil((xMax - xMin) / 5);
    for (let x = Math.floor(xMin / xStep) * xStep; x <= xMax; x += xStep) {
      const xPos = padding.left + (x - xMin) * xScale;
      ctx.beginPath();
      ctx.moveTo(xPos, padding.top);
      ctx.lineTo(xPos, canvas.height - padding.bottom);
      ctx.stroke();

      // X軸のラベル
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(x.toString(), xPos, canvas.height - padding.bottom / 2);
    }

    // Y軸のグリッド線
    const yStep = Math.ceil((yMax - yMin) / 5);
    for (let y = Math.floor(yMin / yStep) * yStep; y <= yMax; y += yStep) {
      const yPos = canvas.height - padding.bottom - (y - yMin) * yScale;
      ctx.beginPath();
      ctx.moveTo(padding.left, yPos);
      ctx.lineTo(canvas.width - padding.right, yPos);
      ctx.stroke();

      // Y軸のラベル
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(y.toString(), padding.left - 10, yPos + 4);
    }

    // 軸のラベルを描画
    ctx.fillStyle = '#374151';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // 散布図のポイントを描画
    ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
    data.forEach(point => {
      const x = padding.left + (point.x - xMin) * xScale;
      const y = canvas.height - padding.bottom - (point.y - yMin) * yScale;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // 回帰線を描画
    const x1 = xMin;
    const y1 = regressionLine.slope * x1 + regressionLine.intercept;
    const x2 = xMax;
    const y2 = regressionLine.slope * x2 + regressionLine.intercept;

    const plotX1 = padding.left + (x1 - xMin) * xScale;
    const plotY1 = canvas.height - padding.bottom - (y1 - yMin) * yScale;
    const plotX2 = padding.left + (x2 - xMin) * xScale;
    const plotY2 = canvas.height - padding.bottom - (y2 - yMin) * yScale;

    ctx.strokeStyle = 'rgba(220, 38, 38, 0.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(plotX1, plotY1);
    ctx.lineTo(plotX2, plotY2);
    ctx.stroke();

    // R²値を表示
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`R² = ${regressionLine.r2.toFixed(2)}`, canvas.width - padding.right, padding.top + 20);
  }, [data, regressionLine, xLabel, yLabel]);

  return (
    <div className="w-full">
      <canvas
        ref={canvasRef}
        width={800}
        height={height}
        className="w-full h-auto"
      />
    </div>
  );
});

export { ScatterPlotChart };