import React, { useState, useEffect } from 'react';
import { useReportGenerator, ReportStatus, ReportConfig } from '@/hooks/useReportGenerator';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Download, Plus, Loader2, AlertCircle, RefreshCw, Trash2 } from 'lucide-react';
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { format } from 'date-fns';

const ReportsPage: React.FC = () => {
  const [pastReports, setPastReports] = useState<ReportStatus[]>([]);
  const [listLoading, setListLoading] = useState<boolean>(false);
  const [listError, setListError] = useState<Error | null>(null);

  const {
    generateReport,
    cancelReport,
    deleteReport,
    fetchPastReports,
    currentReport,
    loading: generatorLoading,
    error: generatorError
  } = useReportGenerator();

  useEffect(() => {
    const loadReports = async () => {
      setListLoading(true);
      setListError(null);
      try {
        const reports = await fetchPastReports();
        setPastReports(reports);
      } catch (err) {
        setListError(err instanceof Error ? err : new Error('過去のレポートの読み込みに失敗しました。'));
        console.error("Failed to fetch past reports:", err);
      } finally {
        setListLoading(false);
      }
    };
    loadReports();
  }, [fetchPastReports]);

  const handleGenerateNewReport = async () => {
    const reportConfig: ReportConfig = {
      title: `カスタムレポート ${format(new Date(), 'yyyy-MM-dd HH:mm')}`,
      reportType: 'custom' as const
    };
    try {
    } catch (err) {
      console.error("Report generation request failed:", err);
    }
  };

  const handleDeleteReport = async (reportId: string) => {
     if (!window.confirm('本当にこのレポートを削除しますか？')) {
       return;
     }
     try {
        await deleteReport(reportId);
        setPastReports(prev => prev.filter(report => report.id !== reportId));
     } catch (err) {
        console.error("Failed to delete report:", err);
        alert(`レポートの削除に失敗しました: ${err instanceof Error ? err.message : '不明なエラー'}`);
     }
  };

  const renderCurrentReportStatus = () => {
    if (!currentReport) return null;

    const isProcessing = currentReport.status === 'pending' || currentReport.status === 'processing';
    const isCompleted = currentReport.status === 'completed';
    const isError = currentReport.status === 'error';
    const progress = currentReport.progress ?? 0;

    return (
      <Card className="mb-6 border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-900/20">
        <CardHeader>
          <CardTitle className="text-blue-800 dark:text-blue-300">現在のレポート生成状況</CardTitle>
        </CardHeader>
        <CardContent>
           <p className="font-medium mb-2">{currentReport.title || 'レポート'}</p>
          {isProcessing && (
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>生成中...</span>
              <Progress value={progress} className="w-full h-2" />
              <span>{progress}%</span>
            </div>
          )}
          {isCompleted && (
            <div className="text-green-700 dark:text-green-400 flex items-center justify-between">
              <span>生成が完了しました。</span>
              {currentReport.url && (
                <Button asChild variant="outline" size="sm">
                  <a href={currentReport.url} target="_blank" rel="noopener noreferrer">
                    <Download className="mr-2 h-4 w-4" />
                    表示/ダウンロード
                  </a>
                </Button>
              )}
            </div>
          )}
          {isError && (
            <div className="text-red-700 dark:text-red-400">
              エラーが発生しました: {currentReport.error || '不明なエラー'}
            </div>
          )}
           <p className="text-xs text-muted-foreground mt-2">
             開始日時: {currentReport.createdAt ? format(new Date(currentReport.createdAt), 'yyyy/MM/dd HH:mm') : 'N/A'}
           </p>
        </CardContent>
        {isProcessing && (
          <CardFooter>
            <Button variant="outline" size="sm" onClick={() => cancelReport(currentReport.id)} disabled={generatorLoading}>
              キャンセル
            </Button>
          </CardFooter>
        )}
      </Card>
    );
  };

  const renderPastReports = () => {
    if (listLoading) {
      return (
        <div className="flex justify-center items-center py-8">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <p className="ml-2 text-muted-foreground">レポート履歴を読み込み中...</p>
        </div>
      );
    }

    if (listError) {
      return (
        <Alert variant="destructive" className="mb-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>履歴読み込みエラー</AlertTitle>
          <AlertDescription>
            {listError.message}
            <Button onClick={() => window.location.reload()} variant="outline" size="sm" className="ml-4">再試行</Button>
          </AlertDescription>
        </Alert>
      );
    }

    if (pastReports.length === 0) {
      return (
        <div className="text-center text-muted-foreground py-8 border border-dashed rounded-lg">
          利用可能な過去のレポートはありません。
        </div>
      );
    }

    return (
       <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
         {pastReports.map((report) => (
           <Card key={report.id} className="flex flex-col">
             <CardHeader>
               <CardTitle className="text-base font-semibold truncate">{report.title || '無題のレポート'}</CardTitle>
               <p className="text-xs text-muted-foreground">
                 {report.createdAt ? format(new Date(report.createdAt), 'yyyy/MM/dd HH:mm') : '日付不明'}
               </p>
             </CardHeader>
             <CardContent className="flex-grow">
                <div className="flex items-center justify-between text-sm mb-2">
                   <span>ステータス:</span>
                   <span className={`font-medium ${
                     report.status === 'completed' ? 'text-green-600 dark:text-green-400' :
                     report.status === 'error' ? 'text-red-600 dark:text-red-400' :
                     'text-yellow-600 dark:text-yellow-400'
                   }`}>
                     {report.status === 'completed' ? '完了' :
                      report.status === 'error' ? 'エラー' :
                      report.status === 'processing' ? `処理中 (${report.progress ?? 0}%)` :
                      '待機中'}
                   </span>
                </div>
                 {report.status === 'error' && (
                    <p className="text-xs text-red-600 dark:text-red-400 truncate">エラー: {report.error || '不明'}</p>
                 )}
             </CardContent>
             <CardFooter className="flex justify-end gap-2">
               {report.status === 'completed' && report.url && (
                 <Button asChild variant="outline" size="sm">
                   <a href={report.url} target="_blank" rel="noopener noreferrer">
                     <Download className="mr-1 h-4 w-4" />
                     表示
                   </a>
                 </Button>
               )}
               <Button
                 variant="ghost"
                 size="sm"
                 onClick={() => handleDeleteReport(report.id)}
                 disabled={generatorLoading}
               >
                 <Trash2 className="h-4 w-4 text-red-500" />
               </Button>
             </CardFooter>
           </Card>
         ))}
       </div>
    );
  };

  const isGeneratingActive = !!currentReport && (currentReport.status === 'pending' || currentReport.status === 'processing');
  const disableGenerateButton = generatorLoading || isGeneratingActive;

  return (
    <div className="container mx-auto py-8 px-4">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
          <h1 className="text-3xl font-bold">
            レポート
          </h1>
          <Button onClick={handleGenerateNewReport} disabled={disableGenerateButton}>
             {generatorLoading || isGeneratingActive ? (
                 <Loader2 className="mr-2 h-4 w-4 animate-spin" />
             ) : (
                 <Plus className="mr-2 h-4 w-4" />
             )}
            新規レポート作成
          </Button>
        </div>

        {generatorError && (
          <Alert variant="destructive" className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>レポート生成エラー</AlertTitle>
            <AlertDescription>{generatorError.message}</AlertDescription>
          </Alert>
        )}

        {renderCurrentReportStatus()}

        <div className="mt-8">
            <div className="flex justify-between items-center mb-4 border-b pb-2">
                <h2 className="text-xl font-semibold">過去のレポート</h2>
                <Button variant="ghost" size="sm" onClick={() => {/* Call loadReports() */}} disabled={listLoading}>
                    <RefreshCw className={`h-4 w-4 ${listLoading ? 'animate-spin' : ''}`} />
                </Button>
            </div>
            {renderPastReports()}
        </div>
    </div>
  );
};

export default ReportsPage;