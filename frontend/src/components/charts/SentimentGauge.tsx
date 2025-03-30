import { useRef, useEffect, memo } from 'react';

interface SentimentGaugeProps {
  positive: number;
  negative: number;
  neutral: number;
  size?: number;
}

const SentimentGauge = memo(function SentimentGauge({
  positive,
  negative,
  neutral,
  size = 200
}: SentimentGaugeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // キャンバスのクリア
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 円の中心と半径
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) * 0.8;
    const innerRadius = radius * 0.6;

    // 全体の円の背景（ライトグレー）
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#f1f5f9';
    ctx.fill();

    // ネガティブ、ニュートラル、ポジティブの色
    const negativeColor = '#ef4444';  // 赤
    const neutralColor = '#a3a3a3';   // グレー
    const positiveColor = '#22c55e';  // 緑

    // 円全体を100%とした角度計算
    const negativeAngle = Math.PI * 2 * negative;
    const neutralAngle = Math.PI * 2 * neutral;
    const positiveAngle = Math.PI * 2 * positive;

    // 各セクションの開始角度（-π/2は12時の位置から開始）
    let startAngle = -Math.PI / 2;

    // ネガティブセクション
    if (negative > 0) {
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.arc(centerX, centerY, radius, startAngle, startAngle + negativeAngle);
      ctx.closePath();
      ctx.fillStyle = negativeColor;
      ctx.fill();
      startAngle += negativeAngle;
    }

    // ニュートラルセクション
    if (neutral > 0) {
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.arc(centerX, centerY, radius, startAngle, startAngle + neutralAngle);
      ctx.closePath();
      ctx.fillStyle = neutralColor;
      ctx.fill();
      startAngle += neutralAngle;
    }

    // ポジティブセクション
    if (positive > 0) {
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.arc(centerX, centerY, radius, startAngle, startAngle + positiveAngle);
      ctx.closePath();
      ctx.fillStyle = positiveColor;
      ctx.fill();
    }

    // 中心に白い円を描画（ドーナツグラフにする）
    ctx.beginPath();
    ctx.arc(centerX, centerY, innerRadius, 0, Math.PI * 2);
    ctx.fillStyle = 'white';
    ctx.fill();

    // 感情スコアを中央に表示
    const sentimentScore = positive - negative; // -1から1の範囲

    // スコアテキスト
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 24px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(sentimentScore.toFixed(2), centerX, centerY - 15);

    // ラベルテキスト
    ctx.fillStyle = '#6b7280';
    ctx.font = '14px sans-serif';
    ctx.fillText('感情スコア', centerX, centerY + 15);

    // 各セクションの割合をパーセンテージで表示
    const fontSize = 12;
    ctx.font = `${fontSize}px sans-serif`;

    // ポジティブテキスト（上部）
    const positivePercentage = Math.round(positive * 100);
    ctx.fillStyle = positiveColor;
    ctx.textAlign = 'center';
    ctx.fillText(
      `ポジティブ: ${positivePercentage}%`,
      centerX,
      centerY - radius - 10
    );

    // ネガティブテキスト（左下）
    const negativePercentage = Math.round(negative * 100);
    ctx.fillStyle = negativeColor;
    ctx.textAlign = 'right';
    ctx.fillText(
      `ネガティブ: ${negativePercentage}%`,
      centerX - radius / 2,
      centerY + radius + 10
    );

    // ニュートラルテキスト（右下）
    const neutralPercentage = Math.round(neutral * 100);
    ctx.fillStyle = neutralColor;
    ctx.textAlign = 'left';
    ctx.fillText(
      `ニュートラル: ${neutralPercentage}%`,
      centerX + radius / 2,
      centerY + radius + 10
    );

  }, [positive, negative, neutral, size]);

  return (
    <div className="flex justify-center items-center">
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="max-w-full h-auto"
      />
    </div>
  );
});

export { SentimentGauge };