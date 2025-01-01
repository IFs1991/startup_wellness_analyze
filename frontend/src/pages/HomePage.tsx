import { DashboardCard } from '@/components/dashboard/DashboardCard';
import { useNavigate } from 'react-router-dom';

const dashboardCards = [
  {
    title: '分析概要',
    description: 'インタラクティブなチャートとフィルターでデータを分析',
    buttonText: '分析を表示',
    path: '/analysis'
  },
  {
    title: 'レポート生成',
    description: '複数のフォーマットでレポートにアクセス・ダウンロード',
    buttonText: 'レポートを表示',
    path: '/reports'
  },
  {
    title: '分析設定',
    description: '分析パラメータとAI設定を構成',
    buttonText: '設定する',
    path: '/settings'
  },
];

export function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {dashboardCards.map((card) => (
        <DashboardCard
          key={card.title}
          title={card.title}
          description={card.description}
          buttonText={card.buttonText}
          onClick={() => navigate(card.path)}
        />
      ))}
    </div>
  );
}