import { useRef, useEffect, memo } from 'react';
import { Card } from '@/components/ui/card';

interface SurvivalPoint {
  time: number;
  probability: number;
}

export interface SurvivalCurve {
  name: string;
  data: SurvivalPoint[];
  color: string;
}

interface SurvivalCurveChartProps {
  curves: SurvivalCurve[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
}

const SurvivalCurveChart = memo(function SurvivalCurveChart({
  curves,
  height = 400,
  xLabel = '時間（月）',
  yLabel = '生存確率'
}: SurvivalCurveChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // キャンバスのクリア
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // パディングの設定
    const padding = { top: 40, right: 120, bottom: 60, left: 60 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    // データの最大値と最小値を計算
    const allTimes = curves.flatMap(curve => curve.data.map(point => point.time));
    const allProbabilities = curves.flatMap(curve => curve.data.map(point => point.probability));

    const minTime = Math.min(...allTimes);
    const maxTime = Math.max(...allTimes);
    const minProbability = Math.min(...allProbabilities);
    const maxProbability = 1.0; // 生存確率は通常1.0から始まる

    // スケーリング関数
    const timeToX = (time: number) => padding.left + ((time - minTime) / (maxTime - minTime)) * chartWidth;
    const probabilityToY = (prob: number) => padding.top + chartHeight - ((prob - minProbability) / (maxProbability - minProbability)) * chartHeight;

    // タイトルを描画
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('生存曲線分析', canvas.width / 2, 20);

    // グリッド線を描画
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 0.5;

    // X軸のグリッド線と目盛り
    const timeStep = Math.ceil((maxTime - minTime) / 12);
    for (let t = 0; t <= maxTime; t += timeStep) {
      const x = timeToX(t);

      // グリッド線
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartHeight);
      ctx.stroke();

      // 目盛りラベル
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(t.toString(), x, padding.top + chartHeight + 15);
    }

    // Y軸のグリッド線と目盛り
    const probStep = 0.1;
    for (let p = 0; p <= maxProbability; p += probStep) {
      const y = probabilityToY(p);

      // グリッド線
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();

      // 目盛りラベル
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText((p * 100).toFixed(0) + '%', padding.left - 10, y + 4);
    }

    // 軸ラベルを描画
    ctx.fillStyle = '#374151';
    ctx.font = '12px sans-serif';

    // X軸ラベル
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, padding.left + chartWidth / 2, canvas.height - 15);

    // Y軸ラベル
    ctx.save();
    ctx.translate(15, padding.top + chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // 各曲線を描画
    curves.forEach(curve => {
      const { data, color } = curve;

      // 線を描画
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      data.forEach((point, i) => {
        const x = timeToX(point.time);
        const y = probabilityToY(point.probability);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      // 各ポイントにマーカーを描画
      ctx.fillStyle = color;
      data.forEach(point => {
        const x = timeToX(point.time);
        const y = probabilityToY(point.probability);

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    });

    // 凡例を描画
    const legendX = padding.left + chartWidth + 20;
    const legendY = padding.top;
    const legendLineHeight = 25;

    ctx.font = 'bold 12px sans-serif';
    ctx.fillStyle = '#374151';
    ctx.textAlign = 'left';
    ctx.fillText('グループ', legendX, legendY);

    curves.forEach((curve, i) => {
      const y = legendY + (i + 1) * legendLineHeight;

      // 凡例の色のサンプル
      ctx.fillStyle = curve.color;
      ctx.beginPath();
      ctx.rect(legendX, y - 8, 15, 3);
      ctx.fill();

      // 凡例のテキスト
      ctx.fillStyle = '#374151';
      ctx.font = '12px sans-serif';
      ctx.fillText(curve.name, legendX + 25, y);
    });

  }, [curves, height, xLabel, yLabel]);

  return (
    <Card className="p-4">
      <div className="w-full">
        <canvas
          ref={canvasRef}
          width={800}
          height={height}
          className="w-full h-auto"
        />
      </div>
    </Card>
  );
});

export { SurvivalCurveChart };