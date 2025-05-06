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

export const DescriptiveStatsAnalysis = memo(function DescriptiveStatsAnalysis() {
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<string>("3m");
  const [activeTab, setActiveTab] = useState<string>("summary");
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
    let visualizationType = 'stats_summary'; // Default
    if (activeTab === 'distribution') {
      visualizationType = 'stats_distribution';
    } else if (activeTab === 'outliers') {
      visualizationType = 'stats_outliers';
    } else if (activeTab === 'comparison') {
      visualizationType = 'stats_comparison';
    }

    // Replace 'mockAnalysisResults' with actual results if available
    const mockAnalysisResults = { /* Replace with actual analysis results */ };
    visualizeAnalyzerResults(
      'descriptive_stats', // Analyzer type
      mockAnalysisResults, // Analysis results data
      visualizationType, // Visualization type based on tab
      { timeFrame: selectedTimeFrame } // Options
    );

    return () => {
      resetAll();
    };
  }, [selectedTimeFrame, activeTab, visualizeAnalyzerResults, resetAll]);

  const handleRefresh = () => {
    let visualizationType = 'stats_summary';
    if (activeTab === 'distribution') visualizationType = 'stats_distribution';
    else if (activeTab === 'outliers') visualizationType = 'stats_outliers';
    else if (activeTab === 'comparison') visualizationType = 'stats_comparison';

    const mockAnalysisResults = { /* Replace with actual analysis results */ };
    visualizeAnalyzerResults('descriptive_stats', mockAnalysisResults, visualizationType, { timeFrame: selectedTimeFrame });
  };

  const handleDownload = () => {
    if (chartData?.chart_id) {
      const url = getChartDownloadUrl(chartData.chart_id);
      window.open(url, '_blank');
    }
  };

  const renderContent = (minHeight = '350px') => {
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
       // If raw chart data is returned, render it (e.g., as a table or custom component)
       // return <YourStatsTable data={chartData.chart_data} />;
       return <div className="rounded-md bg-background-main p-4 text-center text-text-muted" style={{ minHeight }}>統計データを表示するコンポーネントが必要です。</div>;
    }

    return <div className="rounded-md bg-background-main p-4 text-center text-text-muted" style={{ minHeight }}>表示するデータがありません。</div>;
  };

  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">記述統計</h2>
          <p className="text-sm text-text-secondary">平均、中央値、標準偏差などの基本的な統計量を計算します。</p>
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
          <TabsTrigger value="summary">基本統計量</TabsTrigger>
          <TabsTrigger value="distribution">分布分析</TabsTrigger>
          <TabsTrigger value="outliers">外れ値分析</TabsTrigger>
          <TabsTrigger value="comparison">企業間比較</TabsTrigger>
        </TabsList>

        <TabsContent value="summary" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>基本統計量</CardTitle>
              <CardDescription>各指標の基本的な統計情報</CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent('350px')}
            </CardContent>
          </Card>
          {/* Remove mock statistics sections */}
          {/*
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>中心傾向</CardTitle>
                <CardDescription>データの中心的な値を示す統計量</CardDescription>
              </CardHeader>
              <CardContent>
                 // Mock data removed
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>ばらつき</CardTitle>
                <CardDescription>データのばらつきを示す統計量</CardDescription>
              </CardHeader>
              <CardContent>
                 // Mock data removed
              </CardContent>
            </Card>
          </div>
          */}
        </TabsContent>

        <TabsContent value="distribution" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>分布分析</CardTitle>
              <CardDescription>データの分布特性の詳細分析</CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent('450px')}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="outliers" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>外れ値分析</CardTitle>
              <CardDescription>異常値や外れ値の検出と分析</CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent('450px')}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="comparison" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>企業間比較</CardTitle>
              <CardDescription>選択した企業間の統計量比較</CardDescription>
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