title: "Startup Wellness データ分析システム追加要件 v2.0"
version: 2.0
date: "2025-03-07"
purpose: "VCの投資判断を包括的にサポートするための機能拡張とUI改善"

background:
  current_system:
    version: 1.3
    focus: "健康施策の効果測定とROI計算のための因果推論"
  limitations:
    - "総合的な財務分析機能の不足"
    - "市場・競合分析の欠如"
    - "チーム・組織評価の欠如"
    - "製品・技術評価の欠如"
    - "複雑な統計情報の直感的理解が困難"

new_functional_requirements:
  financial_analysis_module:
    overview: "投資先企業の包括的な財務状況と成長性を評価する機能"
    features:
      cash_burn_analysis:
        - "月次/四半期ごとのキャッシュバーン率計算"
        - "ランウェイ（資金枯渇までの期間）予測"
        - "業界別ベンチマーク比較"
      unit_economics_analysis:
        - "CAC（顧客獲得コスト）計算と推移分析"
        - "LTV（顧客生涯価値）予測モデル"
        - "LTV/CAC比率の計算と推移グラフ"
        - "MRR/ARR成長率分析"
      growth_metrics:
        - "MoM/QoQ/YoYの自動計算"
        - "T2D3フレームワーク達成度評価"
        - "Rule of 40の達成度計算"
      funding_efficiency:
        - "資金調達ラウンドごとのバリュエーション推移"
        - "希薄化率の計算と分析"
        - "競合他社と比較した資金効率"
    data_sources:
      - source: "月次決算報告"
        processing: "構造化パース、時系列変換"
      - source: "会計システム"
        processing: "予測モデル適用"
      - source: "CRMシステム"
        processing: "コホート分析"
    implementation: |
      class FinancialAnalyzer:
          def calculate_burn_rate(self, financial_data, period='monthly'):
              """キャッシュバーン率とランウェイの計算"""
              pass

          def analyze_unit_economics(self, revenue_data, customer_data, cost_data):
              """LTV/CAC分析の実行"""
              pass

          def evaluate_growth_metrics(self, financial_data, benchmark_data=None):
              """成長指標の評価と業界比較"""
              pass

  market_competitive_module:
    overview: "投資先企業の市場ポジションと競争環境を分析する機能"
    features:
      market_analysis:
        - "TAM/SAM/SOMの評価"
        - "市場成長率の予測"
        - "市場シェア推定と推移分析"
      competitive_positioning:
        - "2軸マッピング"
        - "レーダーチャートによる多次元比較"
        - "差別化要素の可視化と定量評価"
      competitor_tracking:
        - "競合の資金調達情報の自動収集"
        - "製品アップデート・市場投入情報の追跡"
        - "採用動向・チーム拡大の監視"
      market_trends:
        - "業界キーワードの検索トレンド分析"
        - "ソーシャルメディアでの言及分析"
        - "規制環境の変化の影響評価"
    data_sources:
      - source: "調査レポート、業界データ"
        processing: "テキスト抽出、構造化"
      - source: "ニュース、プレスリリース、求人情報"
        processing: "NLP、センチメント分析"
      - source: "Google Trends API"
        processing: "時系列分析"
    implementation: |
      class MarketAnalyzer:
          def estimate_market_size(self, market_data, growth_factors):
              """TAM/SAM/SOM推定"""
              pass

          def create_competitive_map(self, competitor_data, dimensions):
              """競合マッピングの生成"""
              pass

          def track_market_trends(self, keyword_list, date_range):
              """市場トレンドの追跡と分析"""
              pass

  team_organization_module:
    overview: "経営陣とチームの質、組織的な強みとリスクを評価する機能"
    features:
      founding_team_evaluation:
        - "創業者の過去の実績スコアリング"
        - "経営陣のスキルカバレッジ分析"
        - "産業経験・ドメイン知識評価"
      organization_growth:
        - "組織構造の適切性評価"
        - "従業員成長率と離職率の分析"
        - "部門別の人員配置効率"
      talent_acquisition:
        - "トップタレント獲得の成功率"
        - "採用速度と質のバランス分析"
        - "競合との人材獲得競争状況"
      culture_engagement:
        - "従業員満足度・エンゲージメントスコア"
        - "企業文化の強さと一貫性の評価"
        - "リーダーシップ効果性の測定"
    data_sources:
      - source: "LinkedIn、企業サイト"
        processing: "NLP、キャリア軌跡分析"
      - source: "組織図、従業員データ"
        processing: "ネットワーク分析"
      - source: "採用システム、求人情報"
        processing: "時系列分析、効率評価"
    implementation: |
      class TeamAnalyzer:
          def evaluate_founding_team(self, founder_profiles, company_stage):
              """創業チームの強み・弱み分析"""
              pass

          def analyze_org_growth(self, employee_data, timeline):
              """組織成長の健全性評価"""
              pass

          def measure_culture_strength(self, engagement_data, survey_results):
              """文化的一貫性と強さの定量化"""
              pass

  product_technology_module:
    overview: "製品競争力と技術的優位性を客観的に評価する機能"
    features:
      product_maturity:
        - "製品開発ステージの客観的評価"
        - "製品ロードマップの実行率分析"
        - "技術的負債の定量評価"
      tech_differentiation:
        - "特許分析（数、質、カバレッジ）"
        - "技術スタックの先進性評価"
        - "R&D効率の測定"
      user_value:
        - "NPS追跡"
        - "ユーザーエンゲージメント指標"
        - "顧客フィードバックの分析"
      innovation_metrics:
        - "新機能リリース頻度と採用率"
        - "特許・知的財産ポートフォリオ成長"
        - "技術投資の収益性評価"
    data_sources:
      - source: "製品ロードマップ"
        processing: "マイルストーン分析"
      - source: "特許データベース"
        processing: "テキストマイニング、引用分析"
      - source: "アナリティクスツール"
        processing: "行動パターン分析"
    implementation: |
      class ProductAnalyzer:
          def assess_product_maturity(self, product_data, market_feedback):
              """製品成熟度の定量評価"""
              pass

          def evaluate_tech_differentiation(self, tech_stack, patent_data, competitor_tech):
              """技術的差別化度の計算"""
              pass

          def analyze_user_engagement(self, analytics_data, feedback_data):
              """ユーザーエンゲージメントの深層分析"""
              pass

  scenario_analysis_module:
    overview: "多様な将来シナリオ下での投資先企業のパフォーマンスを予測する機能"
    features:
      monte_carlo_simulation:
        - "複数パラメータの確率的変動に基づくシミュレーション"
        - "1,000回以上の試行による分布生成"
        - "信頼区間付き予測結果の可視化"
      scenario_modeling:
        - "ベース/ブル/ベアケースのシナリオ定義"
        - "カスタム条件の定義と保存機能"
        - "シナリオ間の結果比較可視化"
      sensitivity_analysis:
        - "主要変数の感度テスト"
        - "トルネードチャートによる影響度表示"
        - "クリティカルパラメータの特定"
      survivability_analysis:
        - "資金枯渇確率の計算"
        - "追加資金調達必要性の予測"
        - "最低サバイバル条件の特定"
    data_sources:
      - source: "財務予測モデル"
        processing: "確率モデル変換"
      - source: "変動パラメータ"
        processing: "確率分布フィッティング"
      - source: "マクロ経済指標"
        processing: "相関分析、回帰分析"
    implementation: |
      class ScenarioAnalyzer:
          def run_monte_carlo(self, model_params, num_simulations=10000):
              """モンテカルロシミュレーションの実行"""
              pass

          def define_scenario(self, name, parameter_set, assumptions):
              """カスタムシナリオの定義"""
              pass

          def perform_sensitivity_analysis(self, model, parameters, ranges):
              """感度分析とトルネードチャート生成"""
              pass

  integrated_dashboard:
    overview: "すべての分析結果を一元的に表示し、投資判断に必要な総合的な視点を提供"
    features:
      company_overview:
        - "主要KPIのスコアカード表示"
        - "時系列トレンドグラフ"
        - "投資判断に関わる主要指標のヒートマップ"
      portfolio_comparison:
        - "複数企業の横断比較"
        - "成長ステージ別のベンチマーク比較"
        - "ポートフォリオ全体の健全性スコア"
      investment_decision_support:
        - "投資先企業の総合スコアリング表示"
        - "リスク・リターンマトリクス"
        - "追加投資判断のための意思決定ツリー"
      custom_reports:
        - "LP向け、内部向けなど目的別テンプレート"
        - "エクスポート機能（PDF, Excel, PPT）"
        - "定期自動レポート配信設定"
    implementation: |
      class DashboardController:
          def generate_company_overview(self, company_id, time_range):
              """企業概況ダッシュボードの生成"""
              pass

          def create_portfolio_comparison(self, company_ids, metrics):
              """ポートフォリオ比較ビューの作成"""
              pass

          def generate_custom_report(self, template_id, parameters, format='pdf'):
              """カスタムレポートの生成と出力"""
              pass

ui_design:
  approach: "プログレッシブ・ディスクロージャー（段階的開示）"
  layers:
    summary_layer:
      audience: "全ユーザー向け"
      visual_elements:
        scorecards:
          size: "240px × 120px"
          content: "中央に大きな数値/評価、下部に簡潔なラベル、前回比の変化指標"
        gauge_meters:
          scale: "0-100"
          colors: "赤→黄→緑のグラデーション"
          features: "業界平均値のマーカー表示"
        impact_badges:
          format: "A+, A, B+, B, C+, C, D形式のグレーディング"
          variants: "「優秀」「良好」「要改善」などの単語評価"
      guidelines:
        - "数値は常に「良い/悪い」の文脈付きで表示"
        - "専門用語の代わりに一般的な表現を使用"
        - "数字は丸めて表示（小数点以下は必要な場合のみ）"
        - "複雑な指標は単一スコアに統合"

    interpretation_layer:
      audience: "関心のあるユーザー向け"
      visual_elements:
        trend_graphs:
          timespan: "最低6か月の時系列"
          features: "主要トレンドライン、ベンチマーク表示、簡易解釈テキスト"
        comparison_charts:
          content: "競合/業界平均との棒グラフ比較、複数指標の横並び"
          features: "百分位数マーカー表示"
        correlation_maps:
          type: "2D散布図"
          features: "相関性の強調表示、主要クラスターのグルーピング"
      guidelines:
        - "主要な統計指標は平易な解説付きで表示"
        - "信頼区間は「確信度」として表現"
        - "因果関係と相関関係の区別を視覚的に明示"
        - "ユーザー操作可能なフィルターと範囲選択"

    expert_layer:
      audience: "専門家向け"
      visual_elements:
        statistical_tables:
          content: "完全な統計値表示"
          features: "ソート・フィルター機能、データエクスポート機能"
        advanced_visualizations:
          types: ["ボックスプロット", "バイオリンプロット", "ヒートマップ", "ネットワークグラフ"]
        model_details:
          content: "モデルパラメータ表示、残差プロット、モデル診断情報"
          features: "アルゴリズム選択オプション"
      guidelines:
        - "生データへのアクセス提供"
        - "モデルのすべてのパラメータとメタデータを表示"
        - "手法の限界と前提条件を明示"
        - "分析の再現・検証を可能にする情報提供"

  component_implementations:
    causal_inference_results:
      summary_layer: "効果スコア（0-100）、確信度（★）、シンプルな前後比較"
      interpretation_layer: "時系列グラフ、効果の解釈、信頼区間の視覚化"
      expert_layer: "完全な統計レポート、モデル比較、感度分析結果"

    roi_calculation:
      summary_layer: "ROI値（%）、回収期間、投資判断信号（赤/黄/緑）"
      interpretation_layer: "ROI構成要素のブレイクダウン、シナリオ比較チャート"
      expert_layer: "確率分布グラフ、感度分析、詳細計算過程"

    market_position:
      summary_layer: "市場ポジションスコア、競争力バッジ、成長潜在性評価"
      interpretation_layer: "競合マッピング、市場シェア推移、差別化要素視覚化"
      expert_layer: "完全な競合分析、テキストマイニング結果、特許分析詳細"

tech_stack:
  frontend_extensions:
    core:
      - "React Component拡張"
      - "D3.js"
      - "Material UI"
      - "Redux"
    visualization:
      - "Recharts"
      - "ECharts"
      - "Vega-Lite"
      - "Nivo"

  backend_extensions:
    data_processing:
      - "PySpark"
      - "Scikit-learn (既存)"
      - "PyMC/Stan (既存)"
      - "NetworkX (既存)"
      - "Gensim"
      - "NLTK/spaCy"
      - "PyTorch"
    api_integration:
      - "FastAPI (既存)"
      - "GraphQL"
      - "Airflow"
      - "Celery"

  data_storage_extensions:
    structured:
      - "Google Cloud SQL (既存)"
      - "BigQuery"
      - "ClickHouse"
    unstructured:
      - "Firestore (既存)"
      - "Elasticsearch"
      - "Google Cloud Storage"

  external_integrations:
    data_sources:
      - "CrunchBase API"
      - "LinkedIn API"
      - "CB Insights API"
      - "PitchBook API"
      - "Google Trends API"
      - "News API"
    analysis_tools:
      - "Jupyter Hub"
      - "Google Data Studio"
      - "Tableau"

data_model_extensions:
  market_competitive_model: |
    CREATE TABLE market_data (
      market_id SERIAL PRIMARY KEY,
      industry_id INTEGER REFERENCES industries(id),
      market_name VARCHAR(255) NOT NULL,
      market_size_current BIGINT,
      market_size_projected BIGINT,
      cagr DECIMAL(5,2),
      data_source VARCHAR(255),
      last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE competitor_profiles (
      competitor_id SERIAL PRIMARY KEY,
      company_name VARCHAR(255) NOT NULL,
      founded_date DATE,
      headcount INTEGER,
      funding_total BIGINT,
      last_funding_date DATE,
      last_funding_amount BIGINT,
      last_funding_type VARCHAR(50),
      business_model VARCHAR(100),
      market_id INTEGER REFERENCES market_data(market_id),
      website_url VARCHAR(255),
      description TEXT,
      last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

  financial_model: |
    CREATE TABLE financial_metrics (
      metric_id SERIAL PRIMARY KEY,
      company_id INTEGER REFERENCES companies(company_id),
      reporting_date DATE NOT NULL,
      revenue BIGINT,
      gross_profit BIGINT,
      ebitda BIGINT,
      net_income BIGINT,
      cash_balance BIGINT,
      burn_rate DECIMAL(10,2),
      runway_months INTEGER,
      cac DECIMAL(10,2),
      ltv DECIMAL(10,2),
      mrr DECIMAL(10,2),
      arr DECIMAL(10,2),
      growth_rate_mom DECIMAL(5,2),
      growth_rate_yoy DECIMAL(5,2),
      unit_economics_score INTEGER,
      data_source VARCHAR(100),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

  firestore_collections:
    market_intelligence: |
      {
        market_name: String,
        description: String,
        market_reports: Array<{
          report_date: Timestamp,
          report_source: String,
          key_findings: String,
          growth_projections: Map<String, Number>,
          trends: Array<String>
        }>,
        news_mentions: Array<{
          date: Timestamp,
          source: String,
          title: String,
          url: String,
          sentiment: Number,
          relevance: Number
        }>,
        updated_at: Timestamp
      }

    scenarios: |
      {
        name: String,
        description: String,
        created_by: String,
        created_at: Timestamp,
        updated_at: Timestamp,
        company_id: String,
        parameters: Map<String, {
          base_value: Number,
          min_value: Number,
          max_value: Number,
          distribution: String,
          distribution_params: Map<String, Number>
        }>,
        results: {
          summary: {
            expected_outcome: Number,
            confidence_interval: {
              lower: Number,
              upper: Number
            },
            success_probability: Number
          },
          simulation_data: Array<Number>,
          sensitivity_analysis: Map<String, Number>
        }
      }

implementation_roadmap:
  phase1:
    name: "基盤拡張"
    duration: "8週間"
    tasks:
      - "データモデル拡張の実装"
      - "外部データソース連携API実装"
      - "UI層別アプローチの基本フレームワーク構築"
      - "主要コンポーネントの開発"

  phase2:
    name: "コア機能実装"
    duration: "12週間"
    tasks:
      - "財務分析モジュール実装"
      - "市場・競合分析モジュール実装"
      - "チーム・組織分析モジュール実装"
      - "製品・技術評価モジュール実装"

  phase3:
    name: "高度分析機能実装"
    duration: "10週間"
    tasks:
      - "シナリオ分析・ストレステストモジュール実装"
      - "統合ダッシュボード実装"
      - "レポート生成機能拡張"
      - "API拡張とインテグレーション"

  phase4:
    name: "UI/UX最適化とテスト"
    duration: "6週間"
    tasks:
      - "層別UIの詳細実装と最適化"
      - "ユーザビリティテスト"
      - "パフォーマンス最適化"
      - "セキュリティテスト"

  phase5:
    name: "統合・展開"
    duration: "4週間"
    tasks:
      - "既存システムとの統合テスト"
      - "ユーザーマニュアル・ガイド作成"
      - "パイロットユーザーへの展開"
      - "フィードバック収集と調整"

non_functional_requirements:
  performance:
    - "ダッシュボード初期表示: 3秒以内"
    - "データ更新反映: リアルタイム（Firestore）/ 1時間以内（バッチ処理）"
    - "シナリオ分析実行: 1,000回のシミュレーションを30秒以内"
    - "レポート生成: 60秒以内"

  security:
    - "複数ポートフォリオ間のデータ分離"
    - "外部データソースアクセスの安全な認証情報管理"
    - "競合情報アクセスの厳格な権限コントロール"
    - "匿名化とデータ保護の強化"

  usability:
    - "初回ユーザーガイド（インタラクティブチュートリアル）"
    - "コンテキスト依存ヘルプ機能"
    - "ユーザー習熟度に基づく自動UI調整"
    - "主要分析パターンのテンプレート化"

  scalability:
    - "分析モジュールのプラグイン型アーキテクチャ"
    - "カスタムデータソース接続フレームワーク"
    - "ユーザー定義指標の追加機能"
    - "APIファーストアプローチによる外部連携性確保"

expected_outcomes:
  - outcome: "投資判断の質向上"
    description: "健康施策ROIだけでなく、投資判断のすべての側面を考慮した総合的評価の実現"

  - outcome: "意思決定時間の短縮"
    description: "データ収集・分析の自動化により、投資判断プロセスが最大60%迅速化"

  - outcome: "リスク評価の精緻化"
    description: "シナリオ分析とストレステストによる投資リスクの正確な定量化"

  - outcome: "ユーザー満足度向上"
    description: "層別UIアプローチにより、技術的知識レベルに関わらず全ユーザーが価値を獲得"

  - outcome: "持続的競争優位性"
    description: "高度な分析機能とUI設計による他VCとの差別化とポートフォリオパフォーマンス向上"