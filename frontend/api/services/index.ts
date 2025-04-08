import AnalysisService from './AnalysisService';
import DashboardService from './DashboardService';
import ReportService from './ReportService';
import StartupDataService from './StartupDataService';

export {
  AnalysisService,
  DashboardService,
  ReportService,
  StartupDataService,
};

// APIサービスを一括でエクスポート
export default {
  analysis: AnalysisService,
  dashboard: DashboardService,
  reports: ReportService,
  startupData: StartupDataService,
};