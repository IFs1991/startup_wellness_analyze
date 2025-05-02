import React from "react";
import { BaseAnalysisTemplate } from "./base-analysis-template";
import type { BaseAnalysisTemplateProps } from "@/types/analysis-template";

export interface CompanyInfo {
  id: string;
  name: string;
  description?: string;
  [key: string]: any;
}

export interface AnalysisPageTemplateProps {
  company: CompanyInfo;
  analyses: {
    key: string;
    title: string;
    description?: string;
    templateProps: BaseAnalysisTemplateProps;
  }[];
  customHeader?: React.ReactNode;
  customFooter?: React.ReactNode;
}

/**
 * 企業ごとに複数の分析法ページを動的生成するテンプレート
 * analyses配列に各分析法のテンプレートpropsを渡すことで自動レンダリング
 */
export const CompanyAnalysisPageTemplate: React.FC<AnalysisPageTemplateProps> = ({
  company,
  analyses,
  customHeader,
  customFooter
}) => {
  return (
    <div className="company-analysis-page-root">
      {customHeader}
      <header className="company-analysis-header">
        <h1 className="company-analysis-title">{company.name} の分析</h1>
        {company.description && <p className="company-analysis-description">{company.description}</p>}
      </header>
      <main className="company-analysis-main">
        {analyses.map(analysis => {
          // title, descriptionをtemplatePropsから除外
          const { title, description, templateProps } = analysis;
          const { title: _t, description: _d, ...restTemplateProps } = templateProps || {};
          return (
            <section key={analysis.key} className="company-analysis-section">
              <BaseAnalysisTemplate
                title={title}
                description={description}
                {...restTemplateProps}
              />
            </section>
          );
        })}
      </main>
      {customFooter}
    </div>
  );
};