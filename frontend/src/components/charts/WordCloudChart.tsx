import { useRef, useEffect, memo } from 'react';
import { Card } from '@/components/ui/card';

interface WordCloudItem {
  text: string;
  value: number;
}

interface WordCloudChartProps {
  data: WordCloudItem[];
  width?: number;
  height?: number;
}

const WordCloudChart = memo(function WordCloudChart({
  data,
  width = 600,
  height = 400
}: WordCloudChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // キャンバスのクリア
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // データの最大値と最小値を取得して正規化するための準備
    const maxValue = Math.max(...data.map(item => item.value));
    const minValue = Math.min(...data.map(item => item.value));
    const valueRange = maxValue - minValue;

    // 単語の配置領域を計算
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const maxRadius = Math.min(centerX, centerY) * 0.9;

    // 単語の位置と角度をランダムに決定（重複回避のため）
    const placedWords: Array<{
      text: string;
      x: number;
      y: number;
      width: number;
      height: number;
      fontSize: number;
    }> = [];

    // 単語をサイズでソート（大きいものから配置）
    const sortedData = [...data].sort((a, b) => b.value - a.value);

    // 各単語を描画
    for (const item of sortedData) {
      // 単語のサイズを値に基づいて計算（正規化して12〜48pxの範囲に）
      const normalizedValue = (item.value - minValue) / valueRange;
      const fontSize = 12 + normalizedValue * 36;

      // フォントを設定
      ctx.font = `${Math.floor(fontSize)}px sans-serif`;

      // 色を設定（値に基づいて色相を変化）
      const hue = 200 + normalizedValue * 160; // 200〜360（青〜赤）
      const saturation = 70 + normalizedValue * 30; // 70〜100%
      const lightness = 45 + normalizedValue * 10; // 45〜55%
      ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;

      // テキストのサイズを計測
      const metrics = ctx.measureText(item.text);
      const textWidth = metrics.width;
      const textHeight = fontSize;

      // 単語の位置を決定（中心から放射状に配置し、重複を避ける）
      let placed = false;
      let attempts = 0;
      const maxAttempts = 100;

      while (!placed && attempts < maxAttempts) {
        // 角度と半径をランダムに決定
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * maxRadius;

        // 位置を計算
        const x = centerX + Math.cos(angle) * radius - textWidth / 2;
        const y = centerY + Math.sin(angle) * radius + textHeight / 4;

        // 他の単語と重なっていないか確認
        let overlaps = false;
        for (const placedWord of placedWords) {
          if (
            x < placedWord.x + placedWord.width &&
            x + textWidth > placedWord.x &&
            y - textHeight < placedWord.y &&
            y > placedWord.y - placedWord.height
          ) {
            overlaps = true;
            break;
          }
        }

        // 重なっていなく、キャンバス内に収まる場合は配置
        if (!overlaps &&
            x >= 0 &&
            x + textWidth <= canvas.width &&
            y - textHeight >= 0 &&
            y <= canvas.height) {
          ctx.fillText(item.text, x, y);
          placedWords.push({
            text: item.text,
            x,
            y,
            width: textWidth,
            height: textHeight,
            fontSize
          });
          placed = true;
        }

        attempts++;
      }

      // 最大試行回数を超えた場合は、重なりを許容して配置
      if (!placed) {
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * maxRadius * 0.8;
        const x = centerX + Math.cos(angle) * radius - textWidth / 2;
        const y = centerY + Math.sin(angle) * radius + textHeight / 4;

        // キャンバスからはみ出さないように調整
        const adjustedX = Math.max(0, Math.min(canvas.width - textWidth, x));
        const adjustedY = Math.max(textHeight, Math.min(canvas.height, y));

        ctx.fillText(item.text, adjustedX, adjustedY);
      }
    }

  }, [data, width, height]);

  return (
    <Card className="p-4">
      <div className="w-full flex justify-center">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="max-w-full h-auto"
        />
      </div>
    </Card>
  );
});

export { WordCloudChart };