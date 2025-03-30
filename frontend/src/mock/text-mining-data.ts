// テキストマイニング用のモックデータ
export const mockTextMiningData = {
  wordCloud: [
    { text: 'ワークライフバランス', value: 85 },
    { text: 'リモートワーク', value: 78 },
    { text: '健康', value: 72 },
    { text: 'ストレス', value: 68 },
    { text: '満足度', value: 65 },
    { text: 'メンタルヘルス', value: 63 },
    { text: '柔軟性', value: 58 },
    { text: 'コミュニケーション', value: 55 },
    { text: '生産性', value: 53 },
    { text: '福利厚生', value: 50 },
    { text: '運動', value: 47 },
    { text: '研修', value: 45 },
    { text: '休暇', value: 43 },
    { text: '家族', value: 41 },
    { text: '食事', value: 38 },
    { text: '睡眠', value: 36 },
    { text: '残業', value: 34 },
    { text: '通勤', value: 31 },
    { text: '評価', value: 29 },
    { text: '時短', value: 27 },
    { text: '報酬', value: 25 },
    { text: '成長', value: 23 },
    { text: '環境', value: 21 },
    { text: '目標', value: 19 },
    { text: '支援', value: 17 },
    { text: '協力', value: 15 },
  ],
  sentimentAnalysis: {
    positive: 0.68,
    negative: 0.15,
    neutral: 0.17
  },
  topicModeling: [
    {
      topic: 'ワークライフバランス',
      keywords: ['時短', '柔軟性', '休暇', '家族', 'リモートワーク'],
      documentCount: 246,
      sentiment: 0.72
    },
    {
      topic: '健康促進',
      keywords: ['運動', '食事', '睡眠', '健康', 'メンタルヘルス'],
      documentCount: 187,
      sentiment: 0.65
    },
    {
      topic: '職場環境',
      keywords: ['コミュニケーション', '環境', '協力', '支援', '評価'],
      documentCount: 154,
      sentiment: 0.58
    },
    {
      topic: 'キャリア開発',
      keywords: ['研修', '成長', '目標', '評価', '報酬'],
      documentCount: 132,
      sentiment: 0.61
    },
    {
      topic: 'リモートワーク',
      keywords: ['通勤', 'リモートワーク', '生産性', '環境', 'コミュニケーション'],
      documentCount: 205,
      sentiment: 0.76
    }
  ],
  insights: [
    'フィードバックの68%がポジティブな感情を示しており、全体的なウェルネス施策の満足度は高い',
    'テキストデータの中で最も頻出するキーワードは「ワークライフバランス」と「リモートワーク」であり、従業員の主要な関心事であることが示唆されている',
    '「リモートワーク」に関するトピックは最も高い感情スコア（0.76）を示しており、従業員の満足度に大きく寄与している',
    '「ストレス」というキーワードは頻出単語の4位であり、継続的な注意が必要な領域である',
    '「健康促進」トピックは2番目に多く言及されているが、感情スコアは中程度（0.65）であり、改善の余地がある'
  ],
  keyTerms: [
    { term: 'ワークライフバランス', occurrence: 246, sentiment: 0.72 },
    { term: 'リモートワーク', occurrence: 231, sentiment: 0.76 },
    { term: '健康促進プログラム', occurrence: 187, sentiment: 0.65 },
    { term: 'メンタルヘルスサポート', occurrence: 175, sentiment: 0.62 },
    { term: '柔軟な勤務時間', occurrence: 168, sentiment: 0.71 }
  ]
};