```yaml
# Startup Wellness データ分析システム 統合要件定義書 v1.3
# 2024年1月制定

system:
  name: "Startup Wellness データ分析システム"
  version: "1.3"
  last_updated: "2024-01"

# 1. システム概要
overview:
  purpose: |
    Startup Wellnessプログラムの効果を分析し、VCに提出する報告書を自動生成するとともに、
    チャットベースでの企業分析機能を提供する。各VCの投資判断に資する情報を提供することで、
    Startup Wellnessプログラムの価値向上を目指す。
    重点: 施術毎に取得するVASスケールデータと損益計算書データの関連分析に注力する。

  target_users:
    - role: "Startup Wellnessの分析チーム"
      description: "プログラムの効果検証、改善点の特定、レポート作成を行う"
      functions: ["データ入力", "処理", "分析", "可視化", "レポート生成"]
    - role: "VC"
      description: "投資先企業の健康状態、プログラムの効果を把握し、投資判断に活用する"
      functions: ["分析結果の閲覧", "レポートのダウンロード", "分析設定のカスタマイズ"]

  update_background: |
    VCの投資判断最適化を目的に、健康施策のROI計算モデルとベイズ推論機能を追加する。
    既存システム（ver1.2）に拡張を行う。

# 2. システムアーキテクチャ
architecture:
  cloud_infrastructure:
    platform: "Google Cloud Platform (GCP)"
    services:
      - name: "Cloud Run"
        purpose: "マイクロサービスのデプロイメント"
      - name: "Cloud Functions"
        purpose: "サーバーレス関数の実行"
      - name: "Firestore"
        purpose: "NoSQLデータストア"
      - name: "Cloud SQL"
        purpose: "リレーショナルデータベース"
      - name: "Cloud Storage"
        purpose: "ファイルストレージ"
      - name: "Cloud Pub/Sub"
        purpose: "メッセージングサービス"
      - name: "Cloud Load Balancing"
        purpose: "負荷分散"
      - name: "Cloud CDN"
        purpose: "コンテンツ配信"
      - name: "Cloud IAM"
        purpose: "アクセス管理"

  data_stores:
    firestore:
      purpose: "NoSQLデータストア"
      data_types:
        - type: "リアルタイムデータ"
          examples: ["チャットメッセージ", "分析結果のキャッシュ", "ユーザーセッション情報"]
          features: ["高速な読み書き", "リアルタイム同期", "柔軟なスキーマ"]
        - type: "スケーラブルなデータ"
          examples: ["アンケート回答 (VASスケールデータ、自由記述データ)", "施術記録", "分析レポート"]
          features: ["大量のデータを効率的に保存", "柔軟なデータ構造", "高いスケーラビリティ"]
      modeling:
        - "ドキュメント指向で、複雑なデータ構造を柔軟に表現"
        - "コレクションとドキュメントの階層構造を利用して、関連データをグループ化"
        - "サブコレクションを利用して、1対多の関係を表現"

    cloud_sql:
      purpose: "リレーショナルデータベース (PostgreSQL)"
      data_types:
        - type: "構造化データ"
          examples: ["ユーザーマスタ", "企業マスタ", "従業員マスタ", "損益計算書データ"]
          features: ["厳密なスキーマ", "トランザクション処理", "複雑なクエリ"]
        - type: "トランザクション要件の高いデータ"
          examples: ["請求情報", "監査ログ"]
          features: ["ACID特性を保証", "データの整合性を重視"]
      modeling:
        - "テーブルとリレーションシップを利用して、構造化されたデータを表現"
        - "正規化されたデータモデルで、データの重複を排除"

  microservices:
    core_services:
      - name: "認証サービス"
        technology: "Cloud Run + Firebase Authentication"
        features: ["シングルサインオン (SSO) 対応", "マルチファクタ認証 (MFA) サポート"]
      - name: "データ収集サービス"
        technology: "Cloud Functions"
        features: ["Google Forms API連携", "HR系SaaS APIコネクタ", "ファイルアップロード処理"]
      - name: "分析サービス"
        technology: "Cloud Run"
        features: ["データ前処理", "統計分析", "機械学習モデル"]
      - name: "レポート生成サービス"
        technology: "Cloud Run"
        features: ["生成AI API統合", "PDF生成", "カスタマイズ機能"]
      - name: "チャットサービス"
        technology: "Cloud Run + Firestore"
        features: ["生成AI API統合", "リアルタイムメッセージング", "コンテキスト管理"]

  external_integrations:
    hr_saas:
      supported_services: ["KING OF TIME", "カオナビ", "SmartHR", "フレクト", "マネーフォワード クラウド勤怠"]
      data_types: ["勤怠情報", "人事情報", "給与情報", "評価情報", "従業員満足度調査結果"]
      integration_methods: ["REST API", "GraphQL API", "Webhook", "SFTP", "OAuth2.0認証"]

    generative_ai:
      purposes:
        - category: "レポート生成"
          details: ["分析結果の自然言語での説明", "インサイトの抽出", "推奨アクションの提案"]
        - category: "チャット形式での企業分析"
          details: ["対話的な分析シナリオの提供", "データに基づく質問応答", "仮説検証支援"]
      features:
        - "企業選択機能"
        - "ドロップダウンによる企業選択"
        - "検索機能"
        - "お気に入り登録"
        - "分析シナリオ選択"
        - "テンプレートシナリオ"
        - "カスタムシナリオ"
        - "過去の分析履歴"

# 3. コア機能要件
core_requirements:
  data_input:
    priority: "★★★"
    components:
      - name: "Google Forms連携"
        purpose: "Google FormsのAPIを使用してアンケート結果を自動取得する"
        details:
          - "施術毎のアンケート: 主にVASスケールを用いた設問"
          - "身体的症状 (肩こり、眼精疲労、腰痛など)"
          - "精神的状態 (集中力、ストレスレベル、睡眠の質など)"
          - "その他 (仕事へのモチベーション、コミュニケーションなど)"
          - "自由記述項目: 分析に役立つ情報を取得"
        implementation: "data_input.py の GoogleFormsConnector クラス"
        storage: "アンケート回答は、Firestoreのドキュメントとして保存。自由記述データは、テキスト形式と構造化データの両方で保存"

      - name: "CSVインポート"
        purpose: "CSVファイルからの財務データ、人事データのインポート"
        details: "スタートアップ企業の損益計算書、従業員数、離職率などのデータを取得"
        implementation: "data_input.py の read_csv_data 関数"
        storage: "損益計算書データは、Cloud SQLのテーブルに保存"

      - name: "外部データ統合"
        purpose: "外部データソース（業界ベンチマーク、経済指標など）の統合"
        details: "スタートアップ企業の分析結果を業界平均と比較"
        implementation: "data_input.py の ExternalDataFetcher クラス"
        storage: "外部データは、Firestoreのドキュメントとして保存"

      - name: "ファイルアップロード"
        purpose: "施術後のアンケートデータ、問診票、損益計算書のアップロード機能"
        details:
          - "複数ファイルの同時アップロードに対応"
          - "ファイル種別ごとにアップロードエリアを分け、ドラッグ&ドロップ操作を可能にする"
          - "アップロード可能なファイル形式: CSV, Excel, PDF"
        implementation: "data_input.py の upload_files 関数"

    ui_ux:
      - "Google Formsのデザインを踏襲したフォーム"
      - "ドラッグ&ドロップエリアとプライマリカラー（#4285F4）のアップロードボタン"
      - "ファイル種別ごとにラベル付きのアップロードエリア"
      - "アップロードされたファイル名はリストで表示"
      - "トグルスイッチとドロップダウンメニューによる外部データソース連携設定"

    ai_support: "データ入力の自動補完、データ形式のチェック"

  data_processing:
    priority: "★★★"
    components:
      - name: "データ前処理と整形"
        purpose: "pandasを使用してデータの前処理と整形を行う"
        details:
          - "VASデータと損益計算書データを結合"
          - "時系列データへの変換"
        implementation: "data_processing.py の DataPreprocessor クラス"
        storage: "FirestoreからVASデータを取得し、Cloud SQLから損益計算書データを取得して結合。結合後のデータは、分析のためにpandas DataFrameに変換"

      - name: "欠損値処理、異常値処理"
        purpose: "欠損値の処理、異常値の検出と処理を行う"
        details:
          - "データクレンジングの詳細なプロセス実装"
          - "データの検証ルールとチェックポイントの設定"
        implementation: "data_processing.py の DataPreprocessor クラス - handle_missing_values メソッド, detect_outliers メソッド"

      - name: "特徴量エンジニアリング"
        purpose: "分析に有効な特徴量を作成する"
        details: "VAS値の変化率や移動平均などを計算"
        implementation: "data_processing.py の FeatureEngineer クラス"

      - name: "データ品質管理プロセス"
        purpose: "データの品質を監視・管理する"
        details:
          - "データの整合性チェック"
          - "データの更新履歴管理"
        implementation: "data_processing.py の DataQualityChecker クラス"

    ai_support: "データの前処理、整形、クレンジング作業の自動化や支援、欠損値の補完、異常値の検出と処理の精度向上、特徴量エンジニアリングの自動化や提案"

  analysis:
    priority: "★★★"
    components:
      - name: "記述統計量算出"
        purpose: "基本的な統計量を算出する"
        details: "データの平均、中央値、標準偏差などを計算し、データの分布を把握"
        implementation: "analysis.py の calculate_descriptive_stats 関数"
        ai_support: "統計量の結果を人間が理解しやすい自然言語で説明、統計量から得られる洞察や示唆を自動生成"

      - name: "相関分析"
        purpose: "VASデータと損益計算書データの相関関係を分析"
        details:
          - "2つの変数の間の関係性を分析し、相関の強さを数値化"
          - "例: 従業員満足度と生産性の関係などを分析"
        implementation: "analysis.py の correlation_analysis 関数"
        ai_support: "相関関係の強弱を自然言語で表現、相関関係から考えられる仮説や解釈を提示"

      - name: "時系列分析"
        purpose: "VASデータと損益計算書データの経時変化を分析"
        details:
          - "時系列データの変化パターンを分析し、将来の傾向を予測"
          - "例: 売上高の推移を分析し、将来の売上高を予測"
        implementation: "analysis.py の TimeSeriesAnalyzer クラス"
        ai_support: "時系列データのトレンドや季節性を自然言語で説明、予測結果の解釈や将来予測に関するリスク分析を提供"

      - name: "クラスタ分析"
        purpose: "類似した特徴を持つスタートアップ企業や従業員をグループ分けする"
        details: "従業員を健康状態やライフスタイルに基づいてグループ分けし、それぞれのグループに適した健康施策を検討"
        implementation: "analysis.py の ClusterAnalyzer クラス"
        ai_support: "クラスタの特徴を自然言語で説明、各クラスタに適した施策や介入方法を提案"

      - name: "主成分分析"
        purpose: "多数のVAS項目から、従業員の健康状態を代表する少数の主成分を抽出する"
        details:
          - "多数の変数を持つデータを、より少ない変数で表現"
          - "例: 多数の健康指標から主要な指標を抽出し、従業員の健康状態を簡潔に把握"
        implementation: "analysis.py の PCAAnalyzer クラス"
        ai_support: "主成分の意味や解釈を自然言語で説明、データの可視化を支援"

      - name: "生存時間分析"
        purpose: "Startup Wellnessプログラム導入前後における、従業員の離職までの時間を比較分析する"
        details:
          - "あるイベントが発生するまでの時間を分析"
          - "例: 従業員が離職するまでの時間や、顧客がサービスを解約するまでの時間を分析し、離職率や解約率を予測"
        implementation: "analysis.py の SurvivalAnalyzer クラス"
        ai_support: "分析結果を人間が理解しやすい自然言語で説明、イベント発生リスクの高い要因を特定"

      - name: "アソシエーション分析"
        purpose: "特定の健康状態と関連性の高い行動や属性を特定する"
        details:
          - "データ項目間の関連性を分析"
          - "例: 特定の健康状態と関連性の高い行動や属性を特定し、効果的な健康施策を立案"
        implementation: "analysis.py の AssociationAnalyzer クラス"
        ai_support: "発見された関連性の解釈や説明を提供、関連性に基づいた施策や介入方法を提案"

      - name: "テキスト分析"
        purpose: "アンケート自由記述項目から有益な情報を抽出する"
        details:
          - "自然言語処理技術を活用"
          - "生成AIを活用した要約やインサイト生成"
          - "特定のキーワードや感情を含む回答を抽出する機能"
        implementation: "analysis.py の TextMiner クラス"
        storage: "テキスト分析の結果は、Firestoreのドキュメントとして保存"

  vc_roi_calculation:
    priority: "★★★"
    formula: "ROI_{VC} = \\frac{(ΔRevenue + ΔValuation) - C_{program}}{C_{investment}} × 100"
    data_elements:
      - item: "ΔRevenue"
        source: "月次売上データ"
        processing: "時系列因果推論"
      - item: "ΔValuation"
        source: "バリュエーションレポート"
        processing: "EV/EBITDA倍率適用"
      - item: "C_{program}"
        source: "請求管理システム"
        processing: "リアルタイム連携"
      - item: "C_{investment}"
        source: "キャピタルコール記録"
        processing: "投資段階別分類"

  bayesian_inference:
    priority: "★★★"
    overview:
      prior: "過去3年間の全ポートフォリオ企業データ"
      posterior_update: "週次でFirestoreの新規データを反映"
      output: "確率的ROI予測区間（95%信用区間付き）"
    implementation:
      code_example: |
        # ベイズ更新プロセス
        def bayesian_update(prior, likelihood):
            posterior = prior * likelihood
            return posterior.normalize()

        # Cloud Functions連携
        trigger_new_data → update_posterior → push_to_dashboard

  portfolio_network_analysis:
    priority: "★★★"
    metrics:
      - name: "エコシステム係数"
        description: "企業間の健康施策波及効果（0-1尺度）"
      - name: "知識移転指数"
        description: "同業種間のベストプラクティス共有度"
    visualization:
      code_example: |
        {
          "mark": "network",
          "encoding": {
            "nodeSize": {"field": "HIEI", "scale": {"range": [5, 30]}},
            "linkColor": {"field": "synergy", "scale": {"scheme": "redblue"}}
          }
        }

  data_model_extension:
    priority: "★★★"
    new_data_structures:
      - collection: "vc_portfolio"
        fields:
          - name: "ecosystem_impact"
            type: "Map"
            description: "ネットワーク効果係数"
      - collection: "investment"
        fields:
          - name: "risk_adjusted_irr"
            type: "Float"
            description: "健康リスク調整後IRR"
    anonymization:
      - "k-匿名化：投資額を範囲値（例：¥50M-¥100M）に一般化"
      - "差分プライバシー：ε=0.1のラプラスノイズ付加"

  vc_dashboard:
    priority: "★★★"
    panels:
      - name: "投資効率"
        metrics: "ROI_{VC}（確率分布表示）"
        update_frequency: "リアルタイム"
      - name: "ポートフォリオ健康"
        metrics: "HIEI指数ランキング"
        update_frequency: "日次"
      - name: "シナリオ分析"
        metrics: "モンテカルロ予測シミュレーター"
        update_frequency: "オンデマンド"
    interactive_features:
      - "投資段階フィルター（シード/A/Bラウンド）"
      - "業種別ベンチマーク比較"
      - "仮想施策適用シミュレーション"

  visualization:
    priority: "★★★"
    components:
      - name: "ダッシュボード作成"
        purpose: "VC向けにカスタマイズ可能なダッシュボードを提供"
        details:
          - "スタートアップ企業の健康状態、プログラムの効果などを可視化"
          - "インタラクティブなグラフ表示とフィルター機能を提供"
          - "VCが自身の関心に基づいてデータを探索できる"
          - "時系列表示機能: 各社の分析結果を時系列順に表示"
        implementation: "visualization.py の DashboardCreator クラス"

      - name: "グラフ生成"
        purpose: "様々な種類のグラフを生成"
        details:
          - "分析結果を可視化"
          - "線グラフ、棒グラフ、散布図など"
        implementation: "visualization.py の GraphGenerator クラス"

      - name: "インタラクティブな可視化"
        purpose: "インタラクティブなグラフを提供"
        details: "ユーザーがグラフを操作することで、詳細なデータを確認可能"
        implementation: "visualization.py の InteractiveVisualizer クラス"

    ai_support: "分析結果に基づいた最適なグラフの種類を提案、グラフのレイアウトや配色を自動調整、グラフに分析結果の解釈や洞察を自動付加、時系列データのトレンド分析や予測の自動化"

    ui_ux:
      - "白背景、角丸、薄い影のカード"
      - "シンプルな線グラフ、棒グラフ、円グラフ"
      - "ドロップダウンメニュー、チェックボックス、スライダーなどを用いたフィルター"
      - "プライマリカラー（#4285F4）の角丸ボタン、印刷ボタン"

  report_generation:
    priority: "★★★"
    components:
      - name: "PDFレポート自動生成"
        purpose: "Pythonライブラリ (ReportLab) を使用してPDFレポートを自動生成する"
        implementation: "report_generation.py の PDFReportGenerator クラス"

      - name: "レポートカスタマイズ"
        purpose: "対象読者（VC、経営陣、従業員）別のレポートカスタマイズ"
        details:
          - "VC向けには、投資判断に役立つVASデータと損益計算書データの関連分析結果を重点的に記述"
          - "生成AIを活用して、データに基づいた分析結果のサマリーや考察をレポートに自動追記"
          - "生成AIを活用し、過去の分析結果を参考にレポートの構成案を提案"
        implementation: "report_generation.py の CustomReportBuilder クラス"

      - name: "インタラクティブなレポーティングツールの実装"
        priority: "★"
        purpose: "レポートの内容をインタラクティブに操作できるようにする"

      - name: "レポートアーカイブ機能"
        purpose: "過去のレポートをポートフォリオ毎に整理・保管し、VCがいつでも閲覧できるようにする"
        details:
          - "レポート生成時に、自動的にアーカイブに追加"
          - "ポートフォリオ名、レポート作成日などで検索可能にする"
          - "UI上でレポートを選択してプレビュー、ダウンロードできるようにする"

    ai_support: "分析結果を元にレポートの構成や内容を自動生成、レポートに分析結果の解釈や洞察を自動付加、対象読者にあわせたレポートの表現やトーンを自動調整"

    ui_ux:
      - "カード形式で表示されたレポートテンプレート (選択中のテンプレートはプライマリカラーで強調)"
      - "PDF、Excel、Googleスプレッドシート、CSV形式のレポートプレビュー"
      - "プライマリカラーのダウンロードボタン、印刷ボタン"

    storage: "生成されたレポートは、Firestoreのドキュメントとして保存"

  prediction_model:
    priority: "★★"
    components:
      - name: "将来パフォーマンス予測"
        purpose: "機械学習モデルを使用した将来パフォーマンス予測"
        details:
          - "スタートアップ企業の将来の売上高、従業員満足度などを予測"
          - "VASデータと損益計算書データを用いた予測モデル構築"
        implementation: "prediction.py の PerformancePredictor クラス"

      - name: "モデルパフォーマンス監視"
        purpose: "予測モデルの精度を監視"
        implementation: "prediction.py の ModelEvaluator クラス"

      - name: "定期的なモデル再訓練"
        purpose: "最新のデータでモデルを再訓練"
        implementation: "prediction.py の ModelEvaluator クラス"

    risks_and_countermeasures:
      - risk: "予測モデルの精度が低い可能性"
        countermeasure: "モデルの精度向上のための継続的な改善、適切なデータの前処理、モデル選択を行う"

    ai_support: "予測結果の解釈や信頼性の評価を支援、予測結果に基づいた意思決定のサポートを提供"

    ui_ux:
      - "テーブル形式で表示されたモデル一覧 (モデル名、精度指標、作成日時など)"
      - "グラフで視覚的に表示されたパフォーマンス指標"
      - "スケジュール設定とプライマリカラーの実行ボタンによる再訓練設定"
      - "印刷ボタン"

    storage: "予測モデルのメタデータは、Firestoreのドキュメントとして保存"

  user_authentication:
    priority: "★★★"
    components:
      - name: "ログイン機能"
        details: "ユーザー名とパスワードによる認証"
        implementation: "backend/auth.py の authenticate_user 関数"

      - name: "ログアウト機能"
        details: "セッションの終了"
        implementation: "backend/auth.py の logout_user 関数"

      - name: "ユーザー登録機能"
        details:
          - "新規ユーザー登録"
          - "メールアドレス、パスワード、ユーザー名など必要な情報を入力"
          - "パスワードはセキュリティのためハッシュ化して保存"
          - "登録完了メールを送信"
        implementation: "backend/auth.py の register_user 関数"

      - name: "パスワードリセット機能"
        details:
          - "パスワードを忘れた場合の対応"
          - "メールアドレスにリセット用のリンクを送信"
          - "リンクをクリックして新しいパスワードを設定"
        implementation: "backend/auth.py の reset_password 関数"

    ui_ux:
      - "ログイン画面はシンプルで見やすいデザイン"
      - "ログイン状態を表示"
      - "ユーザー登録画面では入力項目を分かりやすく表示"
      - "パスワードリセット機能は、ユーザーが迷うことなく操作できるよう、手順を明確に表示"

    storage: "ユーザー情報は、Cloud SQLのusersテーブルに保存"

  vc_analysis_settings:
    priority: "★★★"
    description: "各VCは、自身のアカウントに紐づく分析設定を行うことができる"
    settings:
      - name: "Google Form アンケート設定"
        details:
          - "Google Formの施術後アンケート設問項目を ドロップダウンで複数選択"
          - "体感的変化、健康状態、仕事への影響など、多様なVAS設問項目を用意"
          - "自由記述項目: 分析に役立つ情報を取得"
      - name: "損益計算書データ項目"
        details: "分析対象とする損益計算書の項目を ドロップダウンで複数選択"
      - name: "分析手法"
        details: "相関分析、時系列分析、クラスタ分析、主成分分析などを ドロップダウンで選択"
      - name: "可視化方法"
        details: "グラフの種類、表示項目などを ドロップダウンで選択"
      - name: "生成AI設定"
        details:
          - "生成AI APIキー入力欄: ユーザーはフロントエンドから生成AIのAPIキーを入力し、システム全体で生成AIの機能を使用できるようにする"
          - "利用可能な生成AIモデル一覧: 利用可能な生成AIモデルが一覧表示され、ユーザーは使用するモデルを選択できる"
          - "利用用途別設定: 生成AIをどの機能に使用するか、個別設定できる"

    ui_ux: "各設定項目には、説明文やツールチップを表示し、VCが迷うことなく設定できるよう配慮する。設定内容をプレビュー表示し、VCが設定結果を確認できる機能を提供する"

    storage: "分析設定は、Firestoreのドキュメントとして保存"

  memo_function:
    priority: "★★"
    components:
      - name: "分析結果メモ"
        purpose: "各分析機能の実行結果画面に、分析結果に関するメモを記録できる機能を提供する"
        details:
          - "メモはユーザー毎に保存"
          - "メモ入力エリアを各分析機能の結果表示画面に設置"
          - "VCが分析結果に基づいて考察した内容や、今後の投資戦略に関するメモなどを自由に記録可能にする"
        storage: "メモは、Firestoreのドキュメントとして保存"

  user_data_independence:
    priority: "★★★"
    components:
      - name: "ユーザー毎のログイン機能"
        purpose: "ユーザー毎にシステムへログインできる機能を提供する"
        details:
          - "ユーザーは、自身のIDとパスワードでシステムにログインできる"
          - "ログイン状態はセッション管理により維持される"
          - "ログインユーザーの権限に応じて、アクセス可能な機能やデータが制限される"
        implementation: "backend/auth.py に関連機能を追加"

      - name: "ユーザー毎のデータ独立性"
        purpose: "ユーザー毎に管理できるデータを独立させる機能を提供する"
        details:
          - "各ユーザーがアップロード、作成、分析したデータは、他のユーザーからはアクセスできないように分離される"
          - "データベースレベルで、ユーザーIDをキーとしてデータを分離する"
          - "各ユーザーの分析設定、レポートテンプレート、メモなども独立して管理される"
        implementation:
          - "データベース設計において、ユーザーIDをキーとしてデータを分離する"
          - "各モジュールにおいて、ユーザーIDを考慮したデータアクセス制御を実装する"
          - "Firestoreのコレクションやドキュメントの構造を、ユーザーIDをキーとして分離する"
          - "Cloud SQLのテーブルにおいても、ユーザーIDをキーとしてデータを分離する"

# 4. セキュリティ要件
security_requirements:
  data_security:
    encryption: "データの暗号化 (AES-256) (security.py の DataEncryptor クラス)"
    access_control: "アクセス制御 (OAuth2.0) (security.py の OAuthHandler クラス)"
    data_anonymization:
      implementation: "security.py の DataAnonymizer クラス"
      methods:
        - "k-匿名化"
        - "l-多様化"
        - "t-近接性を組み合わせた手法"
      tools: "ARX Data Anonymization Tool"
      process:
        - "識別子の削除"
        - "準識別子の一般化"
        - "センシティブ属性の多様化"
        - "再識別リスクの評価"
      enterprise_protection: "VCの機密情報保護のため、データベース上では企業名を匿名化し、VCが管理するIDと紐付けることで、個々の企業名を秘匿化する"

    personal_data_protection:
      - policy: "データ最小化"
        description: "必要最小限のデータのみを収集・保持"
      - policy: "目的制限"
        description: "データ使用目的の明確化と遵守"
      - policy: "ストレージ制限"
        description: "不要になったデータの安全な削除"
      - policy: "データ主体の権利保護"
        description: "アクセス権、訂正権、削除権の保証"

    additional_measures:
      - "生成AI API Keyの安全な管理: 環境変数への保存、暗号化"
      - "パスワードはハッシュ化して保存: bcryptなどの強力なハッシュ関数を使用"
      - "ブルートフォース攻撃対策: ログイン試行回数制限"
      - "セッション管理: セキュアなセッション管理を実装"

  enhanced_security:
    additional_measures:
      - measure: "ROI計算式の暗号化保存（AES-256-GCM）"
      - measure: "バリュエーションデータのゼロ知識証明適用"

    access_control:
      - role: "VCアナリスト"
        permissions: "ROI計算式パラメータ調整"
      - role: "ポートフォリオマネージャー"
        permissions: "ネットワーク分析詳細閲覧"
      - role: "LP（投資家）"
        permissions: "集計値のみ表示"

# 5. 非機能要件
non_functional_requirements:
  performance:
    data_processing: "1時間以内に完了すること"
    report_generation: "5分以内に完了すること"
    concurrent_users: "最大100ユーザー"
    response_time: "3秒以内"

  scalability:
    design: "新しい分析手法の追加が容易な設計"
    code_structure: "モジュール化されたコード構造"
    processing_capacity:
      description: "データ量増加に伴う処理能力の拡張計画"
      implementation: "scalability.py の AWSAutoScaler クラス, RDSReplicaManager クラス"
    multiple_startups: "複数のスタートアップを同時に分析する能力 (scalability.py 参照)"
    additional_plans:
      horizontal_scaling: "Cloud Runの自動スケーリング機能を利用"
      vertical_scaling: "必要に応じてCloud SQLインスタンスタイプのアップグレード"
      database_scaling: "Cloud SQL Read Replicasの使用"
      architecture:
        - "マルチテナント・アーキテクチャの採用"
        - "テナント別のデータ分離とアクセス制御"
        - "分散処理フレームワーク（Apache Spark）の導入検討"
      streaming:
        - "Cloud Pub/Subを使用したリアルタイムデータ取り込み"
        - "Apache Flinkを用いたストリーム処理の実装"
        - "リアルタイムダッシュボードの更新機能"

  compliance:
    data_protection:
      - "GDPR、CCPAなどの関連データ保護法への対応"
      - "データ処理の法的根拠の明確化"
      - "データ保護影響評価（DPIA）の実施"
      - "データ主体の権利行使のためのプロセス整備"
      - "データ越境転送の適切な管理"

    industry_specific:
      - industry: "ヘルスケア"
        regulations: ["HIPAA（米国）", "医療情報システムの安全管理に関するガイドライン（日本）の遵守"]
      - industry: "金融"
        regulations: ["PCI DSS", "金融商品取引法（日本）の遵守"]

# 6. システム構成
system_configuration:
  backend:
    - "Python 3.9+"
    - "FastAPI (APIフレームワーク)"
    - "PostgreSQL (データベース)"

  frontend:
    - "Dash by Plotly (ダッシュボード)"

  deployment:
    - "Docker"
    - "GCP (Cloud Run, Cloud SQL, Firestore, etc.)"

# 7. 主要モジュール
main_modules:
  data_acquisition:
    file: "backend/data_input.py"
    features:
      - "Google Forms API連携"
      - "CSVファイル、Excelファイル、Googleスプレッドシートファイル読み込み"
      - "外部データソース統合"
      - "施術後のアンケートデータ、問診票、損益計算書アップロード機能"
    interfaces: ["Google Forms API", "外部データベースAPI"]

  data_processing:
    file: "backend/data_processing.py"
    features:
      - "前処理 (欠損値処理、異常値検出)"
      - "特徴量エンジニアリング"
      - "データ品質管理プロセス"

  analysis:
    file: "backend/analysis.py"
    features:
      - "記述統計算出"
      - "相関分析"
      - "時系列分析"
      - "テキストマイニング"
      - "クラスタ分析"
      - "主成分分析"
      - "生存時間分析"
      - "アソシエーション分析"
      - "高度な分析手法（因果推論、A/Bテストなど）"

  visualization:
    file: "backend/visualization.py"
    features:
      - "ダッシュボード生成"
      - "グラフ作成"

  report_generation:
    file: "backend/report_generation.py"
    features:
      - "PDFレポート自動生成"
      - "カスタマイズされたレポート作成"

  prediction:
    file: "backend/prediction.py"
    features:
      - "機械学習モデル構築"
      - "将来パフォーマンス予測"
      - "モデル評価と改善"

  authentication:
    file: "backend/auth.py"
    features:
      - "ログイン、ログアウト、ユーザー登録、パスワードリセット機能"

  generative_ai:
    file: "backend/generative_ai.py"
    features:
      - "生成AI APIキーの管理"
      - "生成AI APIの呼び出し"
      - "生成AI機能の実行"
      - "生成AIモデルの選択"

# 8. データフロー
data_flow:
  main_flows:
    - "アンケートデータ → データ取得モジュール → データ処理モジュール"
    - "処理済みデータ → 分析モジュール → 可視化モジュール"
    - "分析結果 → レポート生成モジュール → PDF、Excel、Googleスプレッドシート、CSV出力"
    - "処理済みデータ → 予測モジュール → 将来予測結果"
    - "施術後のアンケートデータ、問診票、損益計算書 → データ取得モジュール → データ処理モジュール"
    - "生成AI設定 → 生成AIモジュール"

# 9. データ管理
data_management:
  metadata_management:
    - "データディクショナリの作成と維持"
    - "変数の定義、単位、許容範囲などの詳細な記録"

  version_control:
    - "データセットのバージョン管理方法"
    - "分析モデルのバージョン管理とトラッキング"

  data_governance:
    - "データの所有権と利用権限の明確化"
    - "データアクセスポリシーの詳細"

  backup_and_recovery:
    - "データ損失防止のための具体的な戦略"
    - "緊急時のデータ復旧プロセス"

# 10. 再現性の確保
reproducibility:
  - "分析プロセスの再現性を保証するための手順"
  - "コードとデータの保存方法"

# 11. エラー処理
error_handling:
  data_acquisition_errors: "リトライメカニズム実装"
  analysis_errors: "エラーログ記録、管理者通知"
  report_generation_errors: "バックアップデータを使用した再生成"
  file_upload_errors: "エラーメッセージ表示、ファイル形式チェック"

# 12. テスト計画
testing_plan:
  unit_testing: "pytest使用"
  integration_testing: "モジュール間の連携テスト"
  load_testing: "Apache JMeter使用"
  security_testing:
    - "OWASP ZAPを使用した脆弱性スキャン"
    - "ペネトレーションテスト（年1回）"
    - "セキュリティコード解析（SonarQube使用）"
    - "定期的なセキュリティ監査（四半期ごと）"
    - "インシデント対応訓練（年2回）"
  test_cases: "各機能に対する具体的なテストケースや受け入れ基準を定義する"
  ai_testing: "生成AI APIの出力の正確性、信頼性、倫理性などを検証するテストケースを含める"

# 13. ドキュメント
documentation:
  - "システム設計書"
  - "APIドキュメント (Swagger UI)"
  - "ユーザーマニュアル"
  - "データディクショナリ"

# 14. 開発スケジュール
development_schedule:
  requirements_and_design: "3週間"
  development: "10週間"
  testing: "3週間"
  deployment_and_operations_testing: "2週間"

# 15. 実装ロードマップ
implementation_roadmap:
  phases:
    - phase: "1. ベイズコア開発"
      content: "推論エンジン基盤構築"
      duration: "6週間"
    - phase: "2. データパイプライン拡張"
      content: "バリュエーションデータ連携"
      duration: "3週間"
    - phase: "3. ダッシュボード改修"
      content: "確率分布可視化機能"
      duration: "4週間"
    - phase: "4. セキュリティ強化"
      content: "暗号化モジュール実装"
      duration: "2週間"

# 16. 期待される効果
expected_effects:
  - metric: "投資判断速度"
    improvement: "40%向上（従来比）"
  - metric: "ポートフォリオ企業の健康関連リスク可視化率"
    improvement: "95%達成"
  - metric: "LP向け報告資料作成工数"
    improvement: "70%削減"

# 17. 保守・運用計画
maintenance_and_operations:
  - "週次バックアップ"
  - "月次パフォーマンス分析"
  - "四半期ごとの機能アップデート"
  - "継続的な分析モデルの評価と改善"
  - "生成AI APIのアップデート対応: 生成AI APIのバージョンアップや機能追加に対応するための計画を含める"

# 18. トレーニング計画
training_plan:
  target: ["Startup Wellness分析チーム", "VC"]
  content: ["システム操作説明", "データ分析の基本", "レポート作成方法", "生成AI分析評価機能の使い方"]
  methods: ["オンラインチュートリアル", "ハンズオンセミナー", "FAQページ"]

# 19. データリテラシー向上プログラム
data_literacy_program:
  - "VCや経営陣向けのデータ解釈ワークショップ"
  - "分析結果の効果的な伝達方法のトレーニング"

# 20. 監視・ログ計画
monitoring_and_logging:
  purpose: "システムの安定稼働状況、パフォーマンス、セキュリティ状況を監視し、問題発生時には迅速な対応を行う"

  monitoring_targets:
    - "システムリソース (CPU、メモリ、ディスク使用量など)"
    - "アプリケーションパフォーマンス (レスポンスタイム、エラー発生率など)"
    - "セキュリティイベント (不正アクセス試行、データ漏洩など)"
    - "生成AI APIの利用状況 (APIコール数、レスポンスタイムなど)"

  log_collection: "システムの各コンポーネントからログを収集し、一元的に管理する"
  log_analysis: "収集したログを分析し、システムの稼働状況や問題点の把握、セキュリティ脅威の検知などに役立てる"

  tools:
    monitoring: "Cloud Monitoring"
    log_collection: "Cloud Logging"
    log_analysis: ["Cloud Logging", "Elasticsearch Service"]

  alerts: "監視項目で閾値を超えた場合、管理者にアラートを通知する"

# 21. データ移行計画
data_migration:
  - "既存システムから新システムへのデータ移行手順、スケジュール、責任者などを定義する"
  - "データ移行に伴うリスクとその対策を検討する"

# 22. ステークホルダー分析
stakeholder_analysis:
  - "プロジェクトに関わるステークホルダー（Startup Wellness分析チーム、VC、システム開発チームなど）を特定し、それぞれの要求事項、関心事、影響力を分析する"
  - "ステークホルダーとのコミュニケーション計画を策定する"

# 23. UI/UX
ui_ux:
  design: "全体のデザインはGoogleのマテリアルデザインに準拠"
  colors:
    primary: "#4285F4 (Google Blue)"
    secondary: "#EA4335 (Google Red)"
    background: "#FFFFFF (White)"
    text: "#212121 (Dark Gray)"
    supporting_text: "#757575 (Gray)"
    border: "#EEEEEE (Light Gray)"

  components:
    search_bar: "白背景、角丸、グレーの境界線"
    account_menu: "ユーザーアイコンとドロップダウンメニュー"
    notification: "ベルアイコンで、未読通知がある場合は赤いバッジを表示"
    loading: "プライマリカラーの円形プログレスバー"
    error_message: "赤いテキストで表示"

# 24. 印刷機能
printing:
  - "各画面には印刷ボタンが配置され、表示されている内容を印刷できます"
  - "印刷レイアウトは、画面表示に合わせて最適化されます"
  - "対応ファイル形式: PDF、Excel、Googleスプレッドシート、CSV"

# 25. 生成AI APIキーの利用
generative_ai_key_usage:
  - "システムは、ユーザーがフロントエンドの「分析設定画面」で生成AI APIキーを入力することで、システム全体で生成AIの機能を使用できるようになります"
  - "ユーザーは、「生成AI設定」セクションで、生成AI APIキーの入力、利用可能な生成AIモデルの一覧表示、利用用途別設定（データ入力支援、データ処理、分析、可視化、レポート生成、予測モデル）を行うことができます"

# 26. システム互換性とユニーク機能
compatibility_and_uniqueness:
  compatibility: "本追加要件は既存システム（ver1.2）との完全互換性を保ちます"
  unique_features: "特にベイズ推論とネットワーク分析の組み合わせにより、健康投資がもたらすエコシステム全体への波及効果を世界で初めて定量化可能にします"
```