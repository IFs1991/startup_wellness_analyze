"use client"

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw, AlertCircle } from "lucide-react"
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"
import { useVisualization } from '@/hooks/useVisualization'; // Import the hook
import { Skeleton } from '@/components/ui/skeleton'; // For loading state
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'; // For error state
import Image from 'next/image'; // To display chart image if applicable

export const CorrelationAnalysis = memo(function CorrelationAnalysis() {
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<string>("12m");
  const [activeTab, setActiveTab] = useState<string>("matrix");
  const {
    isLoading,
    error,
    chartData, // Assuming visualizeAnalyzerResults returns chart data or URL
    visualizeAnalyzerResults,
    getChartDownloadUrl,
    resetAll
  } = useVisualization();

  // Fetch data when component mounts or dependencies change
  useEffect(() => {
    let visualizationType = 'corr_matrix'; // Default
    if (activeTab === 'scatter') {
      visualizationType = 'corr_scatter';
    } else if (activeTab === 'partial') {
      visualizationType = 'corr_partial';
    } else if (activeTab === 'time-lag') {
      visualizationType = 'corr_time_lag';
    }

    // Replace 'mockAnalysisResults' with actual results if available
    const mockAnalysisResults = { /* Replace with actual analysis results */ };
    visualizeAnalyzerResults(
      'correlation', // Analyzer type
      mockAnalysisResults, // Analysis results data
      visualizationType, // Visualization type based on tab
      { timeFrame: selectedTimeFrame } // Options
    );

    return () => {
      resetAll();
    };
  }, [selectedTimeFrame, activeTab, visualizeAnalyzerResults, resetAll]);

  const handleRefresh = () => {
    let visualizationType = 'corr_matrix';
    if (activeTab === 'scatter') visualizationType = 'corr_scatter';
    else if (activeTab === 'partial') visualizationType = 'corr_partial';
    else if (activeTab === 'time-lag') visualizationType = 'corr_time_lag';

    const mockAnalysisResults = { /* Replace with actual analysis results */ };
    visualizeAnalyzerResults('correlation', mockAnalysisResults, visualizationType, { timeFrame: selectedTimeFrame });
  };

  const handleDownload = () => {
    if (chartData?.chart_id) {
      const url = getChartDownloadUrl(chartData.chart_id);
      window.open(url, '_blank');
    }
  };

  const renderContent = (minHeight = '400px') => {
    if (isLoading) {
      return <Skeleton className="w-full" style={{ height: minHeight }} />;
    }

    if (error) {
      return (
        <Alert variant="destructive" style={{ minHeight }}>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>エラー</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      );
    }

    if (chartData?.chart_url) {
      return <Image src={chartData.chart_url} alt={`${activeTab} chart`} width={800} height={400} className="w-full h-auto" style={{ minHeight }} />;
    } else if (chartData?.chart_data) {
       // Render raw data (e.g., correlation matrix values)
       // return <YourCorrelationTable data={chartData.chart_data} />;
       return <div className="rounded-md bg-background-main p-4 text-center text-text-muted" style={{ minHeight }}>相関データを表示するコンポーネントが必要です。</div>;
    }

    return <div className="rounded-md bg-background-main p-4 text-center text-text-muted" style={{ minHeight }}>表示するデータがありません。</div>;
  };

  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">相関分析</h2>
          <p className="text-sm text-text-secondary">異なる指標間の関係性を分析します。</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select value={selectedTimeFrame} onValueChange={setSelectedTimeFrame}>
            <SelectTrigger className="w-[120px]">
              <SelectValue placeholder="期間" />
            </SelectTrigger>
            <SelectContent>
              {timeFrameOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon" disabled>
            <Filter className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={handleRefresh} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
          <Button variant="outline" size="icon" onClick={handleDownload} disabled={isLoading || !chartData?.chart_id}>
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-4">
          <TabsTrigger value="matrix">相関行列</TabsTrigger>
          <TabsTrigger value="scatter">散布図</TabsTrigger>
          <TabsTrigger value="partial">偏相関</TabsTrigger>
          <TabsTrigger value="time-lag">時間ラグ相関</TabsTrigger>
        </TabsList>

        <TabsContent value="matrix" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>相関行列ヒートマップ</CardTitle>
              <CardDescription>変数間の相関係数を視覚化</CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent('400px')}
            </CardContent>
          </Card>
          {/* Remove mock data sections */}
          {/*
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>主要相関ペア</CardTitle>
                <CardDescription>強い相関がある変数ペア</CardDescription>
              </CardHeader>
              <CardContent>
                 // Mock data removed
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>相関解釈</CardTitle>
                <CardDescription>相関分析の洞察</CardDescription>
              </CardHeader>
              <CardContent>
                 // Mock data removed
              </CardContent>
            </Card>
          </div>
          */}
        </TabsContent>

        <TabsContent value="scatter" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>散布図マトリックス</CardTitle>
              <CardDescription>変数間の関係を散布図で表示</CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent('450px')}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="partial" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>偏相関分析</CardTitle>
              <CardDescription>他の変数の影響を制御した相関関係</CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent('450px')}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="time-lag" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>時間ラグ相関分析</CardTitle>
              <CardDescription>時間差を考慮した変数間の関係</CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent('450px')}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
})