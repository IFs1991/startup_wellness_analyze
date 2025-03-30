// 関連分析用のモックデータ
export const mockAssociationData = {
  rules: [
    { antecedent: ['エンゲージメント向上施策'], consequent: ['生産性向上'], support: 0.42, confidence: 0.85, lift: 2.1 },
    { antecedent: ['ワークライフバランス改善'], consequent: ['離職率低下'], support: 0.38, confidence: 0.79, lift: 2.4 },
    { antecedent: ['健康促進プログラム'], consequent: ['欠勤率低下'], support: 0.35, confidence: 0.82, lift: 2.5 },
    { antecedent: ['メンタルヘルスサポート'], consequent: ['従業員満足度向上'], support: 0.33, confidence: 0.88, lift: 2.3 },
    { antecedent: ['リモートワーク導入'], consequent: ['ワークライフバランス改善'], support: 0.41, confidence: 0.76, lift: 1.9 },
    { antecedent: ['柔軟な勤務時間'], consequent: ['従業員満足度向上'], support: 0.39, confidence: 0.81, lift: 2.1 },
    { antecedent: ['チームビルディング活動'], consequent: ['コミュニケーション向上'], support: 0.37, confidence: 0.74, lift: 1.8 },
    { antecedent: ['スキル開発プログラム'], consequent: ['イノベーション増加'], support: 0.29, confidence: 0.72, lift: 1.9 },
    { antecedent: ['健康促進プログラム', 'メンタルヘルスサポート'], consequent: ['全体的なウェルネス向上'], support: 0.28, confidence: 0.91, lift: 2.8 },
    { antecedent: ['エンゲージメント向上施策', 'スキル開発プログラム'], consequent: ['収益成長'], support: 0.26, confidence: 0.87, lift: 2.6 }
  ],
  insights: [
    '健康促進プログラムとメンタルヘルスサポートの組み合わせは、全体的なウェルネス向上と非常に強い関連性があります（信頼度91%）',
    'エンゲージメント向上施策を実施している企業は、高い確率（85%）で生産性の向上も経験しています',
    'ワークライフバランス改善施策は、離職率の低下と強い関連性があります（信頼度79%）',
    'メンタルヘルスサポートプログラムは、従業員満足度向上に最も高い信頼度（88%）を示しています',
    'エンゲージメント向上施策とスキル開発プログラムの両方を実施している企業は、高い確率（87%）で収益成長に繋がっています'
  ],
  metrics: {
    totalRules: 28,
    avgSupport: 0.34,
    avgConfidence: 0.79,
    avgLift: 2.1
  }
};