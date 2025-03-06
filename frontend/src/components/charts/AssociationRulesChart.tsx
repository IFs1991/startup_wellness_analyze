import { useRef, useEffect, memo } from 'react';
import { Card } from '@/components/ui/card';

interface Rule {
  antecedent: string[];
  consequent: string[];
  support: number;
  confidence: number;
  lift: number;
}

interface AssociationRulesChartProps {
  rules: Rule[];
  height?: number;
}

const AssociationRulesChart = memo(function AssociationRulesChart({
  rules,
  height = 400
}: AssociationRulesChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // キャンバスのクリア
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // パディングの設定
    const padding = { top: 40, right: 40, bottom: 40, left: 200 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    // データの最大値を計算
    const maxLift = Math.max(...rules.map(rule => rule.lift));
    const maxConfidence = 1.0; // 信頼度は0〜1の範囲

    // ルールごとの高さを計算
    const barHeight = Math.min(30, chartHeight / rules.length);
    const barSpacing = Math.min(10, chartHeight / rules.length / 3);

    // タイトルを描画
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('関連ルールの強さ（信頼度と支持度）', canvas.width / 2, 20);

    // 凡例を描画
    ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
    ctx.fillRect(padding.left, 30, 15, 10);
    ctx.fillStyle = '#374151';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('信頼度', padding.left + 20, 38);

    ctx.fillStyle = 'rgba(220, 38, 38, 0.7)';
    ctx.fillRect(padding.left + 100, 30, 15, 10);
    ctx.fillStyle = '#374151';
    ctx.textAlign = 'left';
    ctx.fillText('リフト値', padding.left + 120, 38);

    // X軸の目盛りを描画
    ctx.strokeStyle = '#e5e7eb';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'center';
    ctx.font = '10px sans-serif';

    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (chartWidth * i) / 10;
      const value = (i / 10) * maxConfidence;

      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartHeight);
      ctx.stroke();

      ctx.fillText(value.toFixed(1), x, padding.top + chartHeight + 15);
    }

    // Y軸を描画
    ctx.fillStyle = '#374151';
    ctx.textAlign = 'left';
    ctx.font = '12px sans-serif';
    ctx.fillText('関連ルール', 10, padding.top - 10);

    // 各ルールを描画
    rules.forEach((rule, i) => {
      const y = padding.top + i * (barHeight + barSpacing);

      // ルールのラベルを描画
      const ruleText = `${rule.antecedent.join(', ')} → ${rule.consequent.join(', ')}`;
      ctx.fillStyle = '#374151';
      ctx.textAlign = 'right';
      ctx.font = '11px sans-serif';

      // テキストを切り詰めて表示
      const maxLabelWidth = padding.left - 10;
      const label = truncateText(ctx, ruleText, maxLabelWidth);
      ctx.fillText(label, padding.left - 10, y + barHeight / 2 + 4);

      // 信頼度のバーを描画
      const confidenceWidth = (rule.confidence / maxConfidence) * chartWidth;
      ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
      ctx.fillRect(padding.left, y, confidenceWidth, barHeight);

      // 信頼度の値を表示
      ctx.fillStyle = '#1f2937';
      ctx.textAlign = 'left';
      ctx.font = 'bold 10px sans-serif';
      ctx.fillText(
        `${(rule.confidence * 100).toFixed(0)}%`,
        padding.left + confidenceWidth + 5,
        y + barHeight / 2 + 3
      );

      // リフト値のマーカーを描画
      const liftX = padding.left + (rule.lift / maxLift) * chartWidth * 0.8; // maxLiftの80%をスケールとして使用
      ctx.fillStyle = 'rgba(220, 38, 38, 0.7)';
      ctx.beginPath();
      ctx.arc(liftX, y + barHeight / 2, 6, 0, Math.PI * 2);
      ctx.fill();

      // リフト値を表示
      ctx.fillStyle = '#1f2937';
      ctx.textAlign = 'left';
      ctx.font = '10px sans-serif';
      ctx.fillText(
        `リフト: ${rule.lift.toFixed(1)}`,
        liftX + 10,
        y + barHeight / 2 + 3
      );

      // 支持度の値を右に表示
      ctx.fillStyle = '#6b7280';
      ctx.textAlign = 'right';
      ctx.font = '10px sans-serif';
      ctx.fillText(
        `支持度: ${(rule.support * 100).toFixed(0)}%`,
        canvas.width - padding.right,
        y + barHeight / 2 + 3
      );
    });

  }, [rules]);

  // テキストを指定された幅に収まるように切り詰める関数
  const truncateText = (ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string => {
    if (ctx.measureText(text).width <= maxWidth) return text;

    let truncated = text;
    while (ctx.measureText(truncated + '...').width > maxWidth && truncated.length > 0) {
      truncated = truncated.substring(0, truncated.length - 1);
    }

    return truncated + '...';
  };

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

export { AssociationRulesChart };