// 生存分析用のモックデータ
export const mockSurvivalData = {
  survivalCurves: [
    {
      name: '全体',
      data: [
        { time: 0, probability: 1.0 },
        { time: 1, probability: 0.97 },
        { time: 2, probability: 0.94 },
        { time: 3, probability: 0.89 },
        { time: 4, probability: 0.85 },
        { time: 5, probability: 0.81 },
        { time: 6, probability: 0.77 },
        { time: 7, probability: 0.74 },
        { time: 8, probability: 0.71 },
        { time: 9, probability: 0.68 },
        { time: 10, probability: 0.65 },
        { time: 11, probability: 0.62 },
        { time: 12, probability: 0.59 },
      ],
      color: '#3b82f6'
    },
    {
      name: 'ウェルネス高',
      data: [
        { time: 0, probability: 1.0 },
        { time: 1, probability: 0.98 },
        { time: 2, probability: 0.96 },
        { time: 3, probability: 0.94 },
        { time: 4, probability: 0.92 },
        { time: 5, probability: 0.90 },
        { time: 6, probability: 0.88 },
        { time: 7, probability: 0.86 },
        { time: 8, probability: 0.84 },
        { time: 9, probability: 0.82 },
        { time: 10, probability: 0.81 },
        { time: 11, probability: 0.79 },
        { time: 12, probability: 0.78 },
      ],
      color: '#22c55e'
    },
    {
      name: 'ウェルネス低',
      data: [
        { time: 0, probability: 1.0 },
        { time: 1, probability: 0.96 },
        { time: 2, probability: 0.90 },
        { time: 3, probability: 0.83 },
        { time: 4, probability: 0.77 },
        { time: 5, probability: 0.70 },
        { time: 6, probability: 0.64 },
        { time: 7, probability: 0.59 },
        { time: 8, probability: 0.54 },
        { time: 9, probability: 0.49 },
        { time: 10, probability: 0.45 },
        { time: 11, probability: 0.41 },
        { time: 12, probability: 0.38 },
      ],
      color: '#ef4444'
    }
  ],
  hazardRates: [
    { timeRange: '0-3ヶ月', hazardRate: 0.03, confident: true },
    { timeRange: '3-6ヶ月', hazardRate: 0.05, confident: true },
    { timeRange: '6-9ヶ月', hazardRate: 0.07, confident: true },
    { timeRange: '9-12ヶ月', hazardRate: 0.08, confident: true },
    { timeRange: '12-15ヶ月', hazardRate: 0.09, confident: false },
    { timeRange: '15-18ヶ月', hazardRate: 0.11, confident: false }
  ],
  medianSurvivalTime: {
    overall: 9.7,
    highWellness: 16.2,
    lowWellness: 5.8
  },
  riskFactors: [
    { factor: '低いウェルネススコア', hazardRatio: 2.87, pValue: 0.001 },
    { factor: '高いストレスレベル', hazardRatio: 2.41, pValue: 0.003 },
    { factor: '低い職場満足度', hazardRatio: 2.12, pValue: 0.005 },
    { factor: 'ワークライフバランスの欠如', hazardRatio: 1.85, pValue: 0.008 },
    { factor: '限られたキャリア成長機会', hazardRatio: 1.63, pValue: 0.012 }
  ],
  insights: [
    'ウェルネススコアが高い従業員は、12ヶ月後の定着率が78%と、低い従業員（38%）と比較して2倍以上高い',
    '最も重要な離職リスク要因は「低いウェルネススコア」で、ハザード比は2.87（p<0.001）と統計的に有意',
    '全体の中央生存時間（定着期間の中央値）は9.7ヶ月だが、ウェルネススコアが高いグループでは16.2ヶ月と大幅に延長',
    '勤続9-12ヶ月の時点でハザード率（離職リスク）が増加し、この期間が介入の重要なタイミングであることを示唆',
    'ウェルネス施策を改善することで、1年間の従業員定着率を最大40%向上させる可能性がある'
  ]
};