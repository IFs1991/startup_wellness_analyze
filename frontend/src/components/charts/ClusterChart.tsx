import { Card, CardContent } from '@/components/ui/card';
import { useEffect, useRef } from 'react';

interface Point {
  x: number;
  y: number;
  clusterId: string;
}

interface ClusterInfo {
  id: string;
  name: string;
  count: number;
  percentage: number;
  color: string;
  features: Record<string, number>;
  insights: string[];
}

interface ClusterChartProps {
  data: Point[];
  clusters: ClusterInfo[];
  selectedCluster: string | null;
  onClusterSelect: (clusterId: string) => void;
  xAxisLabel?: string;
  yAxisLabel?: string;
  height?: number;
}

// キャンバスにクラスター散布図を描画する関数
const drawClusterChart = (
  ctx: CanvasRenderingContext2D,
  data: Point[],
  clusters: ClusterInfo[],
  selectedCluster: string | null,
  width: number,
  height: number,
  xAxisLabel: string = 'エンゲージメント',
  yAxisLabel: string = '満足度'
) => {
  // キャンバスをクリア
  ctx.clearRect(0, 0, width, height);

  // 余白を設定
  const padding = 40;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  // データの最小値、最大値を取得
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);

  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);

  // スケーリング関数
  const xScale = (x: number) => ((x - xMin) / (xMax - xMin)) * plotWidth + padding;
  const yScale = (y: number) => height - (((y - yMin) / (yMax - yMin)) * plotHeight + padding);

  // グリッド線を描画
  ctx.beginPath();
  ctx.strokeStyle = '#e5e5e5';
  ctx.lineWidth = 0.5;

  // 水平グリッド線
  for (let i = 0; i <= 4; i++) {
    const y = yMin + (i / 4) * (yMax - yMin);
    const yPos = yScale(y);

    ctx.moveTo(padding, yPos);
    ctx.lineTo(width - padding, yPos);
  }

  // 垂直グリッド線
  for (let i = 0; i <= 4; i++) {
    const x = xMin + (i / 4) * (xMax - xMin);
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
  ctx.fillText(xAxisLabel, width / 2, height - 10);

  // Y軸ラベル
  ctx.save();
  ctx.translate(15, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText(yAxisLabel, 0, 0);
  ctx.restore();

  // データポイントを描画
  data.forEach(point => {
    const cluster = clusters.find(c => c.id === point.clusterId);
    if (!cluster) return;

    const isSelected = selectedCluster === null || selectedCluster === point.clusterId;

    ctx.beginPath();
    ctx.fillStyle = cluster.color;
    ctx.strokeStyle = isSelected ? '#000' : 'rgba(0, 0, 0, 0.1)';
    ctx.lineWidth = isSelected ? 1 : 0.5;
    ctx.globalAlpha = isSelected ? 1 : 0.3;

    ctx.arc(xScale(point.x), yScale(point.y), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });

  // 不透明度をリセット
  ctx.globalAlpha = 1;

  // クラスターのセントロイドを描画
  clusters.forEach((cluster) => {
    // クラスターに属するポイントのx, y座標の平均を計算（セントロイド）
    const clusterPoints = data.filter(p => p.clusterId === cluster.id);
    if (clusterPoints.length === 0) return;

    const xSum = clusterPoints.reduce((sum, p) => sum + p.x, 0);
    const ySum = clusterPoints.reduce((sum, p) => sum + p.y, 0);
    const xAvg = xSum / clusterPoints.length;
    const yAvg = ySum / clusterPoints.length;

    const isSelected = selectedCluster === null || selectedCluster === cluster.id;
    const centroidSize = isSelected ? 10 : 8;

    // セントロイドを描画
    ctx.beginPath();
    ctx.fillStyle = cluster.color;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.globalAlpha = isSelected ? 1 : 0.5;

    ctx.arc(xScale(xAvg), yScale(yAvg), centroidSize, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // クラスター名ラベルを描画
    ctx.font = isSelected ? 'bold 12px Arial' : '12px Arial';
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.fillText(
      cluster.name.split(':')[0], // クラスター番号のみ表示
      xScale(xAvg),
      yScale(yAvg) - centroidSize - 8
    );
  });

  // 不透明度をリセット
  ctx.globalAlpha = 1;

  // 凡例を描画
  const legendItems = clusters.map((cluster) => ({
    color: cluster.color,
    name: cluster.name.split(':')[0], // クラスター番号のみ表示
    selected: selectedCluster === null || selectedCluster === cluster.id
  }));

  const legendX = padding;
  const legendY = padding;
  const legendWidth = 250;
  const legendHeight = 30 + legendItems.length * 20;

  // 凡例の背景
  ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
  ctx.fillRect(legendX, legendY, legendWidth, legendHeight);
  ctx.strokeStyle = '#ddd';
  ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

  // 凡例タイトル
  ctx.fillStyle = '#333';
  ctx.font = 'bold 12px Arial';
  ctx.textAlign = 'left';
  ctx.fillText('クラスター', legendX + 10, legendY + 20);

  // 凡例アイテム
  legendItems.forEach((item, i) => {
    const itemY = legendY + 40 + i * 20;

    // カラーマーカー
    ctx.beginPath();
    ctx.fillStyle = item.color;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = item.selected ? 1 : 0.5;
    ctx.globalAlpha = item.selected ? 1 : 0.5;
    ctx.arc(legendX + 20, itemY - 5, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // テキスト
    ctx.globalAlpha = item.selected ? 1 : 0.6;
    ctx.fillStyle = '#333';
    ctx.font = item.selected ? 'bold 12px Arial' : '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(item.name, legendX + 35, itemY);
  });

  // 不透明度をリセット
  ctx.globalAlpha = 1;
};

export function ClusterChart({
  data,
  clusters,
  selectedCluster,
  onClusterSelect,
  xAxisLabel = 'エンゲージメント',
  yAxisLabel = '満足度',
  height = 400
}: ClusterChartProps) {
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

    drawClusterChart(
      ctx,
      data,
      clusters,
      selectedCluster,
      rect.width,
      rect.height,
      xAxisLabel,
      yAxisLabel
    );
  }, [data, clusters, selectedCluster, xAxisLabel, yAxisLabel]);

  // クリック時にクラスターを選択する
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // 各クラスターのセントロイドの位置を計算し、クリックした位置がどのクラスターに近いかを判定
    const centroids = clusters.map((cluster) => {
      const clusterPoints = data.filter(p => p.clusterId === cluster.id);
      if (clusterPoints.length === 0) return null;

      const xSum = clusterPoints.reduce((sum, p) => sum + p.x, 0);
      const ySum = clusterPoints.reduce((sum, p) => sum + p.y, 0);
      const xAvg = xSum / clusterPoints.length;
      const yAvg = ySum / clusterPoints.length;

      // 描画上の座標に変換
      const padding = 40;
      const plotWidth = rect.width - padding * 2;
      const plotHeight = rect.height - padding * 2;

      const xValues = data.map(p => p.x);
      const yValues = data.map(p => p.y);

      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);
      const yMin = Math.min(...yValues);
      const yMax = Math.max(...yValues);

      const xScale = (val: number) => ((val - xMin) / (xMax - xMin)) * plotWidth + padding;
      const yScale = (val: number) => rect.height - (((val - yMin) / (yMax - yMin)) * plotHeight + padding);

      return {
        id: cluster.id,
        x: xScale(xAvg),
        y: yScale(yAvg)
      };
    }).filter(c => c !== null) as Array<{ id: string; x: number; y: number }>;

    // クリックした位置から最も近いセントロイドを見つける
    let closestCluster: string | null = null;
    let minDistance = Infinity;

    centroids.forEach(centroid => {
      const distance = Math.sqrt(
        Math.pow(x - centroid.x, 2) + Math.pow(y - centroid.y, 2)
      );

      if (distance < minDistance && distance < 30) { // 30px以内のクリックのみを判定
        minDistance = distance;
        closestCluster = centroid.id;
      }
    });

    if (closestCluster !== null) {
      onClusterSelect(closestCluster);
    }
  };

  return (
    <Card>
      <CardContent className="p-4">
        <div style={{ height: `${height}px`, width: '100%' }}>
          <canvas
            ref={canvasRef}
            className="w-full h-full cursor-pointer"
            onClick={handleCanvasClick}
          />
        </div>
      </CardContent>
    </Card>
  );
}