import AnalysisService from './AnalysisService';
import DashboardService from './DashboardService';
import ReportService from './ReportService';
import StartupDataService from './StartupDataService';
import VisualizationService from './VisualizationService';

export {
  AnalysisService,
  DashboardService,
  ReportService,
  StartupDataService,
  VisualizationService,
};

// APIサービスを一括でエクスポート
export default {
  analysis: AnalysisService,
  dashboard: DashboardService,
  reports: ReportService,
  startupData: StartupDataService,
  visualization: VisualizationService,
};