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

export const PcaAnalysis = memo(function PcaAnalysis() {
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<string>("3m"); // Default might depend on backend
  const [activeTab, setActiveTab] = useState<string>("visualization");
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
    let visualizationType = 'pca_visualization'; // Default
    if (activeTab === 'loadings') {
      visualizationType = 'pca_loadings';
    } else if (activeTab === 'scree') {
      visualizationType = 'pca_scree';
    } else if (activeTab === 'biplot') {
      visualizationType = 'pca_biplot';
    }

    // Replace 'mockAnalysisResults' with actual results if available
    const mockAnalysisResults = { /* Replace with actual analysis results */ };
    visualizeAnalyzerResults(
      'pca', // Analyzer type
      mockAnalysisResults, // Analysis results data
      visualizationType, // Visualization type based on tab
      { timeFrame: selectedTimeFrame } // Options
    );

    return () => {
      resetAll();
    };
  }, [selectedTimeFrame, activeTab, visualizeAnalyzerResults, resetAll]);

  const handleRefresh = () => {
    let visualizationType = 'pca_visualization';
    if (activeTab === 'loadings') visualizationType = 'pca_loadings';
    else if (activeTab === 'scree') visualizationType = 'pca_scree';
    else if (activeTab === 'biplot') visualizationType = 'pca_biplot';

    const mockAnalysisResults = { /* Replace with actual analysis results */ };
    visualizeAnalyzerResults('pca', mockAnalysisResults, visualizationType, { timeFrame: selectedTimeFrame });
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
       // Render raw data (e.g., loadings table)
       // return <YourPcaTable data={chartData.chart_data} />;
       return <div className="rounded-md bg-background-main p-4 text-center text-text-muted" style={{ minHeight }}>PCAデータを表示するコンポーネントが必要です。</div>;
    }

    return <div className="rounded-md bg-background-main p-4 text-center text-text-muted" style={{ minHeight }}>表示するデータがありません。</div>;
  };

  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">主成分分析</h2>
          <p className="text-sm text-text-secondary">データの次元削減を行い、重要な特徴を抽出します。</p>
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
          <TabsTrigger value="visualization">可視化</TabsTrigger>
          <TabsTrigger value="loadings">成分負荷量</TabsTrigger>
          <TabsTrigger value="scree">スクリープロット</TabsTrigger>
          <TabsTrigger value="biplot">バイプロット</TabsTrigger>
        </TabsList>

        <TabsContent value="visualization" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>主成分分析の可視化</CardTitle>
              <CardDescription>2次元平面上での企業分布</CardDescription>
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
                <CardTitle>寄与率</CardTitle>
                <CardDescription>各主成分の説明力</CardDescription>
              </CardHeader>
              <CardContent>
                 // Mock data removed
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>主成分の解釈</CardTitle>
                <CardDescription>主要な主成分の意味</CardDescription>
              </CardHeader>
              <CardContent>
                 // Mock data removed
              </CardContent>
            </Card>
          </div>
          */}
        </TabsContent>

        <TabsContent value="loadings" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>成分負荷量</CardTitle>
              <CardDescription>各変数の主成分への寄与度</CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent('450px')}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="scree" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>スクリープロット</CardTitle>
              <CardDescription>主成分の固有値のプロット</CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent('450px')}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="biplot" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>バイプロット</CardTitle>
              <CardDescription>主成分空間における変数と観測値の関係</CardDescription>
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