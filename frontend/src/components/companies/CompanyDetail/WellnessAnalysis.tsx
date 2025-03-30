import { Card } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { getScoreColor } from '@/lib/utils';

interface WellnessScore {
  overall: number;
  workLife: number;
  stress: number;
  satisfaction: number;
  engagement: number;
}

interface WellnessAnalysisProps {
  scores: WellnessScore;
}

export function WellnessAnalysis({ scores }: WellnessAnalysisProps) {
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold">健康スコア分析</h2>
        <Badge className={getScoreColor(scores.overall)}>
          総合スコア: {scores.overall}
        </Badge>
      </div>

      <div className="space-y-6">
        {Object.entries({
          workLife: 'ワークライフバランス',
          stress: 'ストレスレベル',
          satisfaction: '従業員満足度',
          engagement: 'エンゲージメント',
        }).map(([key, label]) => (
          <div key={key} className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm font-medium">{label}</span>
              <span className="text-sm text-muted-foreground">
                {scores[key as keyof WellnessScore]}%
              </span>
            </div>
            <Progress value={scores[key as keyof WellnessScore]} />
          </div>
        ))}
      </div>
    </Card>
  );
}