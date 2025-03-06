import { useRef, useEffect, memo } from 'react';
import { Card } from '@/components/ui/card';

interface HazardRate {
  timeRange: string;
  hazardRate: number;
  confident: boolean;
}

interface HazardRateChartProps {
  data: HazardRate[];
  height?: number;
}

const HazardRateChart = memo(function HazardRateChart({
  data,
  height = 300
}: HazardRateChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // キャンバスのクリア
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // パディングの設定
    const padding = { top: 40, right: 40, bottom: 60, left: 60 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    // データの最大値を計算
    const maxHazardRate = Math.max(...data.map(item => item.hazardRate)) * 1.2; // 20%余裕を持たせる

    // 棒の幅を計算
    const barWidth = chartWidth / data.length * 0.7;
    const barSpacing = chartWidth / data.length * 0.3;

    // タイトルを描画
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('時間帯別ハザード率', canvas.width / 2, 20);

    // グリッド線を描画
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 0.5;

    // Y軸のグリッド線と目盛り
    const hazardStep = maxHazardRate / 5;
    for (let h = 0; h <= maxHazardRate; h += hazardStep) {
      const y = padding.top + chartHeight - (h / maxHazardRate) * chartHeight;

      // グリッド線
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();

      // 目盛りラベル
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText((h * 100).toFixed(0) + '%', padding.left - 10, y + 4);
    }

    // 各棒を描画
    data.forEach((item, i) => {
      const x = padding.left + (i * (barWidth + barSpacing)) + barSpacing / 2;
      const barHeight = (item.hazardRate / maxHazardRate) * chartHeight;
      const y = padding.top + chartHeight - barHeight;

      // 信頼度に基づいて色を決定
      const color = item.confident ? 'rgba(59, 130, 246, 0.8)' : 'rgba(156, 163, 175, 0.8)';
      const borderColor = item.confident ? 'rgb(37, 99, 235)' : 'rgb(107, 114, 128)';

      // 棒を描画
      ctx.fillStyle = color;
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 1;

      // 角丸の棒を描画
      const radius = 4;
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + barWidth - radius, y);
      ctx.quadraticCurveTo(x + barWidth, y, x + barWidth, y + radius);
      ctx.lineTo(x + barWidth, y + barHeight - radius);
      ctx.quadraticCurveTo(x + barWidth, y + barHeight, x + barWidth - radius, y + barHeight);
      ctx.lineTo(x + radius, y + barHeight);
      ctx.quadraticCurveTo(x, y + barHeight, x, y + barHeight - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();

      ctx.fill();
      ctx.stroke();

      // ハザード率の値を表示
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        (item.hazardRate * 100).toFixed(0) + '%',
        x + barWidth / 2,
        y - 10
      );

      // 時間範囲のラベルを表示
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        item.timeRange,
        x + barWidth / 2,
        padding.top + chartHeight + 20
      );

      // 信頼度が低い場合は注釈を表示
      if (!item.confident) {
        ctx.fillStyle = '#9ca3af';
        ctx.font = 'italic 9px sans-serif';
        ctx.fillText(
          '(予測値)',
          x + barWidth / 2,
          padding.top + chartHeight + 35
        );
      }
    });

    // 軸ラベルを描画
    ctx.fillStyle = '#374151';
    ctx.font = '12px sans-serif';

    // X軸ラベル
    ctx.textAlign = 'center';
    ctx.fillText('時間帯', padding.left + chartWidth / 2, canvas.height - 10);

    // Y軸ラベル
    ctx.save();
    ctx.translate(15, padding.top + chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('ハザード率', 0, 0);
    ctx.restore();

    // 凡例を描画
    const legendX = padding.left + 10;
    const legendY = padding.top + 15;

    // 信頼度の高いデータの凡例
    ctx.fillStyle = 'rgba(59, 130, 246, 0.8)';
    ctx.fillRect(legendX, legendY, 15, 10);
    ctx.strokeStyle = 'rgb(37, 99, 235)';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX, legendY, 15, 10);

    ctx.fillStyle = '#374151';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('実測値', legendX + 25, legendY + 8);

    // 信頼度の低いデータの凡例
    ctx.fillStyle = 'rgba(156, 163, 175, 0.8)';
    ctx.fillRect(legendX + 100, legendY, 15, 10);
    ctx.strokeStyle = 'rgb(107, 114, 128)';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX + 100, legendY, 15, 10);

    ctx.fillStyle = '#374151';
    ctx.font = '10px sans-serif';
    ctx.fillText('予測値', legendX + 125, legendY + 8);

  }, [data, height]);

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

export { HazardRateChart };