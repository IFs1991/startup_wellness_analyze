// 回帰分析用のモックデータ
export const mockRegressionData = {
  wellnessVsRevenue: {
    points: [
      { x: 65, y: 2.1 },
      { x: 68, y: 2.3 },
      { x: 72, y: 2.7 },
      { x: 75, y: 3.0 },
      { x: 78, y: 3.5 },
      { x: 82, y: 4.1 },
      { x: 85, y: 4.8 },
      { x: 88, y: 5.2 },
      { x: 92, y: 5.9 }
    ],
    regressionLine: { slope: 0.084, intercept: -3.42, r2: 0.92 },
    xLabel: 'ウェルネススコア',
    yLabel: '売上成長率 (%)'
  },
  wellnessVsProfit: {
    points: [
      { x: 65, y: 8.2 },
      { x: 68, y: 9.1 },
      { x: 72, y: 10.4 },
      { x: 75, y: 11.8 },
      { x: 78, y: 12.9 },
      { x: 82, y: 14.3 },
      { x: 85, y: 16.7 },
      { x: 88, y: 18.1 },
      { x: 92, y: 21.5 }
    ],
    regressionLine: { slope: 0.495, intercept: -24.1, r2: 0.89 },
    xLabel: 'ウェルネススコア',
    yLabel: '利益率 (%)'
  },
  wellnessVsProductivity: {
    points: [
      { x: 65, y: 72 },
      { x: 68, y: 76 },
      { x: 72, y: 79 },
      { x: 75, y: 83 },
      { x: 78, y: 87 },
      { x: 82, y: 92 },
      { x: 85, y: 96 },
      { x: 88, y: 101 },
      { x: 92, y: 108 }
    ],
    regressionLine: { slope: 1.33, intercept: -13.9, r2: 0.95 },
    xLabel: 'ウェルネススコア',
    yLabel: '生産性指標'
  }
};