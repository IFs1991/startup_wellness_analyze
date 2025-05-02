// 分析テンプレート用共通型定義

export interface AnalysisTab {
  key: string;
  label: string;
  content: React.ReactNode;
}

export interface AnalysisCard {
  title: string;
  description?: string;
  content: React.ReactNode;
}

export interface BaseAnalysisTemplateProps {
  /** タイトル（分析名など） */
  title: string;
  /** サブタイトルや説明文 */
  description?: string;
  /** タブ構造（任意） */
  tabs?: AnalysisTab[];
  /** カード型UI（任意） */
  cards?: AnalysisCard[];
  /** カスタムUI（任意） */
  customContent?: React.ReactNode;
  /** その他拡張props */
  [key: string]: any;
}