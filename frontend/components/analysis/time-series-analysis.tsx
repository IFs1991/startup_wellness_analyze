"use client"

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Download, Filter, RefreshCw, AlertCircle } from "lucide-react"
// import { TimeSeriesChart } from "@/components/charts/time-series-chart" // Assuming this uses mock data, remove or adapt
import { memo } from "react"
import { timeFrameOptions } from "@/lib/constants"
import { useVisualization } from '@/hooks/useVisualization'; // Import the hook
import { Skeleton } from '@/components/ui/skeleton'; // For loading state
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'; // For error state
import Image from 'next/image'; // To display chart image if applicable

export const TimeSeriesAnalysis = memo(function TimeSeriesAnalysis() {
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<string>("3m");
  const [activeTab, setActiveTab] = useState<string>("wellness");
  const {
    isLoading,
    error,
    chartData, // Assuming visualizeAnalyzerResults returns chart data directly or sets it here
    visualizeAnalyzerResults,
    getChartDownloadUrl,
    resetAll
  } = useVisualization();

  // Fetch data when component mounts or dependencies change
  useEffect(() => {
    // Determine visualization type based on active tab
    let visualizationType = 'time_series_wellness'; // Default
    if (activeTab === 'growth') {
      visualizationType = 'time_series_growth';
    } else if (activeTab === 'combined') {
      visualizationType = 'time_series_combined';
    } else if (activeTab === 'comparison') {
      visualizationType = 'time_series_comparison';
    }

    // Example: Fetch data using the hook. Replace 'mockAnalysisResults' with actual results if available
    // You might need to fetch analysis results from another source first.
    const mockAnalysisResults = { /* Replace with actual analysis results */ };
    visualizeAnalyzerResults(
      'time_series', // Analyzer type
      mockAnalysisResults, // Analysis results data
      visualizationType, // Visualization type based on tab
      { timeFrame: selectedTimeFrame } // Options like time frame
    );

    // Cleanup function to reset state when component unmounts or dependencies change
    return () => {
      resetAll();
    };
  }, [selectedTimeFrame, activeTab, visualizeAnalyzerResults, resetAll]);

  const handleRefresh = () => {
    // Re-trigger data fetching
    let visualizationType = 'time_series_wellness';
    if (activeTab === 'growth') visualizationType = 'time_series_growth';
    else if (activeTab === 'combined') visualizationType = 'time_series_combined';
    else if (activeTab === 'comparison') visualizationType = 'time_series_comparison';

    const mockAnalysisResults = { /* Replace with actual analysis results */ };
    visualizeAnalyzerResults('time_series', mockAnalysisResults, visualizationType, { timeFrame: selectedTimeFrame });
  };

  const handleDownload = () => {
    if (chartData?.chart_id) {
      const url = getChartDownloadUrl(chartData.chart_id);
      window.open(url, '_blank');
    }
  };

  const renderContent = () => {
    if (isLoading) {
      return <Skeleton className="h-[400px] w-full" />;
    }

    if (error) {
      return (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>エラー</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      );
    }

    if (chartData?.chart_url) {
      // Assuming the hook returns a URL to the chart image
      return <Image src={chartData.chart_url} alt={`${activeTab} chart`} width={800} height={400} className="w-full h-auto" />;
    } else if (chartData?.chart_data) {
       // If raw chart data is returned, you'd need a component to render it
       // return <YourChartComponent data={chartData.chart_data} />;
       return <p>チャートデータを表示するコンポーネントが必要です。</p>;
    }

    return <p>表示するデータがありません。</p>;
  };

  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">時系列分析</h2>
          <p className="text-sm text-text-secondary">企業のウェルネススコアと成長率の時間的変化を分析します。</p>
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
          <TabsTrigger value="wellness">ウェルネススコア</TabsTrigger>
          <TabsTrigger value="growth">成長率</TabsTrigger>
          <TabsTrigger value="combined">複合指標</TabsTrigger>
          <TabsTrigger value="comparison">企業比較</TabsTrigger>
        </TabsList>

        {/* Render content dynamically based on the active tab */}
        <TabsContent value={activeTab} className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>
                {activeTab === 'wellness' && 'ウェルネススコア推移'}
                {activeTab === 'growth' && '成長率推移'}
                {activeTab === 'combined' && '複合指標分析'}
                {activeTab === 'comparison' && '企業比較'}
              </CardTitle>
              <CardDescription>
                {activeTab === 'wellness' && `過去${timeFrameOptions.find(o => o.value === selectedTimeFrame)?.label || selectedTimeFrame}間の企業ごとのウェルネススコア推移`}
                {activeTab === 'growth' && `過去${timeFrameOptions.find(o => o.value === selectedTimeFrame)?.label || selectedTimeFrame}間の企業ごとの成長率推移`}
                {activeTab === 'combined' && 'ウェルネススコアと成長率の複合指標'}
                {activeTab === 'comparison' && '選択した企業間のウェルネススコア比較'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent()}
            </CardContent>
          </Card>

          {/* Remove mock statistics sections */}
          {/*
          {activeTab === 'wellness' && (
            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>統計サマリー</CardTitle>
                  <CardDescription>ウェルネススコアの統計的概要</CardDescription>
                </CardHeader>
                <CardContent>
                   // Mock data removed
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>変動分析</CardTitle>
                  <CardDescription>ウェルネススコアの変動と傾向</CardDescription>
                </CardHeader>
                <CardContent>
                   // Mock data removed
                </CardContent>
              </Card>
            </div>
          )}
          */}
        </TabsContent>
      </Tabs>
    </div>
  )
})

