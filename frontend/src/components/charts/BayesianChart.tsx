import { Card, CardContent } from '@/components/ui/card';
import { useEffect, useRef } from 'react';

interface Point {
  x: number;
  y: number;
}

interface BayesianChartProps {
  priorDistribution: Point[];
  posteriorDistribution: Point[];
  likelihoodPoints: number[];
}

// キャンバスにベイズ推論のグラフを描画する関数
const drawBayesianChart = (
  ctx: CanvasRenderingContext2D,
  priorDistribution: Point[],
  posteriorDistribution: Point[],
  likelihoodPoints: number[],
  width: number,
  height: number
) => {
  // キャンバスをクリア
  ctx.clearRect(0, 0, width, height);

  // 余白を設定
  const padding = 40;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  // データの最小値、最大値を取得
  const allXValues = [...priorDistribution.map(p => p.x), ...posteriorDistribution.map(p => p.x), ...likelihoodPoints];
  const allYValues = [...priorDistribution.map(p => p.y), ...posteriorDistribution.map(p => p.y)];

  const xMin = Math.min(...allXValues) - 0.05;
  const xMax = Math.max(...allXValues) + 0.05;
  const yMin = 0;
  const yMax = Math.max(...allYValues) * 1.1;

  // スケーリング関数
  const xScale = (x: number) => ((x - xMin) / (xMax - xMin)) * plotWidth + padding;
  const yScale = (y: number) => height - (((y - yMin) / (yMax - yMin)) * plotHeight + padding);

  // グリッド線を描画
  ctx.beginPath();
  ctx.strokeStyle = '#e5e5e5';
  ctx.lineWidth = 0.5;

  // 水平グリッド線
  for (let i = 0; i <= 5; i++) {
    const y = yMin + (i / 5) * (yMax - yMin);
    const yPos = yScale(y);

    ctx.moveTo(padding, yPos);
    ctx.lineTo(width - padding, yPos);
  }

  // 垂直グリッド線
  for (let i = 0; i <= 5; i++) {
    const x = xMin + (i / 5) * (xMax - xMin);
    const xPos = xScale(x);

    ctx.moveTo(xPos, padding);
    ctx.lineTo(xPos, height - padding);
  }

  ctx.stroke();

  // 軸を描画
  ctx.beginPath();
  ctx.strokeStyle = '#666';
  ctx.lineWidth = 1;

  // X軸
  ctx.moveTo(padding, height - padding);
  ctx.lineTo(width - padding, height - padding);

  // Y軸
  ctx.moveTo(padding, height - padding);
  ctx.lineTo(padding, padding);

  ctx.stroke();

  // 軸ラベルを描画
  ctx.font = '12px Arial';
  ctx.fillStyle = '#666';

  // X軸ラベル
  ctx.textAlign = 'center';
  ctx.fillText('効果サイズ', width / 2, height - 10);

  // Y軸ラベル
  ctx.save();
  ctx.translate(15, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('確率密度', 0, 0);
  ctx.restore();

  // X軸の目盛りを描画
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const x = xMin + (i / 5) * (xMax - xMin);
    const xPos = xScale(x);

    ctx.beginPath();
    ctx.moveTo(xPos, height - padding);
    ctx.lineTo(xPos, height - padding + 5);
    ctx.stroke();

    // パーセント表示に変換
    const label = (x * 100).toFixed(0) + '%';
    ctx.fillText(label, xPos, height - padding + 20);
  }

  // 事前分布を描画
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(0, 100, 255, 0.8)';
  ctx.lineWidth = 2;

  priorDistribution.forEach((point, i) => {
    if (i === 0) {
      ctx.moveTo(xScale(point.x), yScale(point.y));
    } else {
      ctx.lineTo(xScale(point.x), yScale(point.y));
    }
  });

  ctx.stroke();

  // 事後分布を描画
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(255, 50, 50, 0.8)';
  ctx.lineWidth = 2;

  posteriorDistribution.forEach((point, i) => {
    if (i === 0) {
      ctx.moveTo(xScale(point.x), yScale(point.y));
    } else {
      ctx.lineTo(xScale(point.x), yScale(point.y));
    }
  });

  ctx.stroke();

  // 観測データポイント（尤度）を描画
  ctx.fillStyle = 'rgba(50, 200, 50, 0.7)';

  likelihoodPoints.forEach(point => {
    ctx.beginPath();
    // 観測データは下部にティックとして表示
    ctx.rect(xScale(point) - 2, height - padding - 5, 4, 10);
    ctx.fill();
  });

  // 凡例を描画
  const legendX = width - padding - 120;
  const legendY = padding + 20;

  ctx.fillStyle = '#333';
  ctx.textAlign = 'left';
  ctx.font = '12px Arial';

  // 事前分布
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(0, 100, 255, 0.8)';
  ctx.lineWidth = 2;
  ctx.moveTo(legendX, legendY);
  ctx.lineTo(legendX + 30, legendY);
  ctx.stroke();
  ctx.fillText('事前分布', legendX + 35, legendY + 4);

  // 事後分布
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(255, 50, 50, 0.8)';
  ctx.lineWidth = 2;
  ctx.moveTo(legendX, legendY + 20);
  ctx.lineTo(legendX + 30, legendY + 20);
  ctx.stroke();
  ctx.fillText('事後分布', legendX + 35, legendY + 24);

  // 観測データ
  ctx.fillStyle = 'rgba(50, 200, 50, 0.7)';
  ctx.beginPath();
  ctx.rect(legendX + 10, legendY + 36, 10, 10);
  ctx.fill();
  ctx.fillStyle = '#333';
  ctx.fillText('観測データ', legendX + 35, legendY + 44);
};

export function BayesianChart({ priorDistribution, posteriorDistribution, likelihoodPoints }: BayesianChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // 高解像度ディスプレイ対応
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // スタイル上のサイズを設定
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    drawBayesianChart(
      ctx,
      priorDistribution,
      posteriorDistribution,
      likelihoodPoints,
      rect.width,
      rect.height
    );
  }, [priorDistribution, posteriorDistribution, likelihoodPoints]);

  return (
    <Card>
      <CardContent className="p-4">
        <div className="h-[300px] w-full">
          <canvas
            ref={canvasRef}
            className="w-full h-full"
          />
        </div>
      </CardContent>
    </Card>
  );
}