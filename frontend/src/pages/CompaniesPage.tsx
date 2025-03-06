import React from 'react';
import { Link } from 'react-router-dom';
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import {
  Building,
  TrendingUp,
  Users,
  ArrowRight,
  Plus
} from 'lucide-react';
import { Company } from '@/types/company';

const CompaniesPage: React.FC = () => {
  // ダミーデータ
  const companies: Company[] = [
    {
      id: '1',
      name: 'テックソリューションズ株式会社',
      industry: 'SaaS',
      wellnessScore: 82,
      employees: 45,
      growthRate: 35
    },
    {
      id: '2',
      name: 'イノベーションラボ合同会社',
      industry: 'フィンテック',
      wellnessScore: 78,
      employees: 28,
      growthRate: 42
    },
    {
      id: '3',
      name: 'フューチャーモビリティ株式会社',
      industry: 'モビリティ',
      wellnessScore: 85,
      employees: 62,
      growthRate: 25
    },
    {
      id: '4',
      name: 'スマートヘルスケア株式会社',
      industry: 'ヘルスケア',
      wellnessScore: 90,
      employees: 76,
      growthRate: 30
    },
    {
      id: '5',
      name: 'グリーンエネルギー株式会社',
      industry: 'クリーンテック',
      wellnessScore: 75,
      employees: 38,
      growthRate: 22
    },
  ];

  // スコアに基づいた色を取得する関数
  const getScoreColor = (score: number) => {
    if (score >= 85) return 'text-green-500';
    if (score >= 75) return 'text-blue-500';
    if (score >= 65) return 'text-amber-500';
    return 'text-red-500';
  };

  // スコアに基づいた背景色を取得する関数
  const getScoreBgColor = (score: number) => {
    if (score >= 85) return 'bg-green-500';
    if (score >= 75) return 'bg-blue-500';
    if (score >= 65) return 'bg-amber-500';
    return 'bg-red-500';
  };

  // スコアに基づいたステータスラベルを取得する関数
  const getScoreStatus = (score: number) => {
    if (score >= 85) return '優秀';
    if (score >= 75) return '良好';
    if (score >= 65) return '平均的';
    return '要改善';
  };

  return (
    <div className="container mx-auto">
      <div className="my-8">
        <div className="flex justify-between items-center mb-8 pb-4 border-b">
          <h1 className="text-3xl font-bold">
            企業一覧
          </h1>
          <Button>
            <Building className="mr-2 h-4 w-4" />
            企業を追加
          </Button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {companies.map((company) => (
            <Card
              key={company.id}
              className="h-full flex flex-col transition-all duration-300 hover:-translate-y-2 hover:shadow-lg relative overflow-visible"
            >
              {/* スコアインジケーター */}
              <div className={`absolute -top-3 right-5 ${getScoreBgColor(company.wellnessScore ?? 0)} text-white rounded-xl px-3 py-1 font-bold text-sm shadow-md z-10`}>
                {company.wellnessScore ?? 0}点
              </div>

              <CardContent className="p-6 flex-grow">
                <div className="flex items-center mb-4">
                  <Avatar className="h-12 w-12 mr-4 bg-primary">
                    <AvatarFallback>{company.name.charAt(0)}</AvatarFallback>
                  </Avatar>
                  <div>
                    <h2 className="text-lg font-bold mb-1">
                      {company.name}
                    </h2>
                    <Badge variant="secondary" className="rounded-sm">
                      {company.industry}
                    </Badge>
                  </div>
                </div>

                <Separator className="my-4" />

                <div className="space-y-4">
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">
                      ウェルネススコア
                    </p>
                    <div className="w-full mt-2">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs">ウェルネススコア</span>
                        <div className="flex items-center gap-1">
                          <Progress value={company.wellnessScore ?? 0} className="h-2" />
                          <div className="text-xs">
                            {getScoreStatus(company.wellnessScore ?? 0)}
                          </div>
                          <span className={`text-xs font-bold ${getScoreColor(company.wellnessScore ?? 0)}`}>
                            {company.wellnessScore ?? 0}/100
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-4">
                    <div className="flex-1 flex items-center">
                      <Users className="h-4 w-4 text-muted-foreground mr-2" />
                      <div>
                        <span className="text-xs text-muted-foreground block">
                          従業員数
                        </span>
                        <span className="text-sm font-medium">
                          {company.employees}名
                        </span>
                      </div>
                    </div>

                    <div className="flex-1 flex items-center">
                      <TrendingUp className="h-4 w-4 text-green-500 mr-2" />
                      <div>
                        <span className="text-xs text-muted-foreground block">
                          成長率
                        </span>
                        <span className="text-sm font-medium text-green-500">
                          +{company.growthRate}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>

              <CardFooter className="p-4 pt-0 justify-end">
                <Button
                  variant="ghost"
                  asChild
                >
                  <Link to={`/companies/${company.id}`} className="font-medium">
                    詳細を見る
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CompaniesPage;