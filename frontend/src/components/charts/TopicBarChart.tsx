import { useRef, useEffect, memo } from 'react';
import { Card } from '@/components/ui/card';

interface Topic {
  topic: string;
  keywords: string[];
  documentCount: number;
  sentiment: number;
}

interface TopicBarChartProps {
  topics: Topic[];
  height?: number;
}

const TopicBarChart = memo(function TopicBarChart({
  topics,
  height = 300
}: TopicBarChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // キャンバスのクリア
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // パディングの設定
    const padding = { top: 20, right: 60, bottom: 60, left: 200 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    // トピック数に基づいて棒の高さを計算
    const barHeight = Math.min(35, chartHeight / topics.length);
    const barSpacing = Math.min(15, chartHeight / topics.length / 3);

    // データの最大文書数を計算
    const maxDocumentCount = Math.max(...topics.map(topic => topic.documentCount));

    // タイトルを描画
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('トピック分析結果', canvas.width / 2, 15);

    // X軸を描画
    ctx.strokeStyle = '#e5e7eb';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'center';
    ctx.font = '10px sans-serif';

    // X軸の目盛りとグリッド線を描画
    const step = Math.ceil(maxDocumentCount / 5);
    for (let i = 0; i <= maxDocumentCount; i += step) {
      const x = padding.left + (i / maxDocumentCount) * chartWidth;

      // グリッド線
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartHeight);
      ctx.stroke();

      // 目盛りラベル
      ctx.fillText(i.toString(), x, padding.top + chartHeight + 15);
    }

    // 各トピックを描画
    topics.forEach((topic, i) => {
      const y = padding.top + i * (barHeight + barSpacing);

      // トピック名を描画
      ctx.fillStyle = '#374151';
      ctx.textAlign = 'right';
      ctx.font = 'bold 12px sans-serif';

      // テキストを切り詰めて表示
      const maxLabelWidth = padding.left - 10;
      const label = truncateText(ctx, topic.topic, maxLabelWidth);
      ctx.fillText(label, padding.left - 10, y + barHeight / 2 + 4);

      // 棒グラフを描画
      const barWidth = (topic.documentCount / maxDocumentCount) * chartWidth;

      // 感情スコアに基づいて色を決定（赤から緑のグラデーション）
      const hue = 0 + topic.sentiment * 120; // 0（赤）から120（緑）
      ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
      ctx.fillRect(padding.left, y, barWidth, barHeight);

      // キーワードを棒の中または右に表示
      ctx.fillStyle = '#1f2937';
      ctx.textAlign = 'left';
      ctx.font = '10px sans-serif';
      const keywordsText = topic.keywords.join(', ');
      const truncatedKeywords = truncateText(ctx, keywordsText, barWidth - 10);

      if (barWidth > 150) {
        // 棒の中に表示できる場合
        ctx.fillStyle = 'white';
        ctx.fillText(truncatedKeywords, padding.left + 10, y + barHeight / 2 + 3);
      } else {
        // 棒の右に表示
        ctx.fillStyle = '#6b7280';
        ctx.fillText(truncatedKeywords, padding.left + barWidth + 10, y + barHeight / 2 + 3);
      }

      // 文書数を表示
      ctx.fillStyle = topic.sentiment > 0.6 ? '#1f2937' : '#f3f4f6';
      ctx.textAlign = 'right';
      ctx.font = 'bold 11px sans-serif';

      if (barWidth > 40) {
        // 棒の中に表示できる場合
        ctx.fillText(
          topic.documentCount.toString(),
          padding.left + barWidth - 10,
          y + barHeight / 2 + 3
        );
      }

      // 感情スコアを右端に表示
      ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
      ctx.textAlign = 'right';
      ctx.font = '11px sans-serif';
      ctx.fillText(
        `感情: ${topic.sentiment.toFixed(2)}`,
        canvas.width - padding.right / 4,
        y + barHeight / 2 + 3
      );
    });

    // X軸ラベルを描画
    ctx.fillStyle = '#374151';
    ctx.textAlign = 'center';
    ctx.font = '12px sans-serif';
    ctx.fillText('文書数', canvas.width / 2, canvas.height - 10);

  }, [topics, height]);

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

export { TopicBarChart };