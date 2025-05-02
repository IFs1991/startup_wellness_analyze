import React from "react";
import type { BaseAnalysisTemplateProps } from "@/types/analysis-template";

/**
 * 分析コンポーネント共通テンプレート
 * 柔軟なpropsで様々な分析UIに対応
 */
export const BaseAnalysisTemplate: React.FC<BaseAnalysisTemplateProps> = ({
  title,
  description,
  tabs,
  cards,
  customContent,
  ...rest
}) => {
  return (
    <div className="analysis-template-root" {...rest}>
      <h2 className="analysis-template-title">{title}</h2>
      {description && <p className="analysis-template-description">{description}</p>}
      {tabs && tabs.length > 0 && (
        <div className="analysis-template-tabs">
          {/* タブUIは必要に応じて実装 */}
          {tabs.map(tab => (
            <div key={tab.key} className="analysis-template-tab">
              <div className="analysis-template-tab-label">{tab.label}</div>
              <div className="analysis-template-tab-content">{tab.content}</div>
            </div>
          ))}
        </div>
      )}
      {cards && cards.length > 0 && (
        <div className="analysis-template-cards">
          {cards.map((card, idx) => (
            <div key={idx} className="analysis-template-card">
              <div className="analysis-template-card-title">{card.title}</div>
              {card.description && <div className="analysis-template-card-description">{card.description}</div>}
              <div className="analysis-template-card-content">{card.content}</div>
            </div>
          ))}
        </div>
      )}
      {customContent && <div className="analysis-template-custom">{customContent}</div>}
    </div>
  );
};