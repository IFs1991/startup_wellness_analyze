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

export const ClusteringAnalysis = memo(function ClusteringAnalysis() {
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<string>("all"); // Default might depend on backend
  const [activeTab, setActiveTab] = useState<string>("kmeans");
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
    let visualizationType = 'cluster_kmeans'; // Default
    if (activeTab === 'hierarchical') {
      visualizationType = 'cluster_hierarchical';
    } else if (activeTab === 'dbscan') {
      visualizationType = 'cluster_dbscan';
    } else if (activeTab === 'evaluation') {
      visualizationType = 'cluster_evaluation';
    }

    // Replace 'mockAnalysisResults' with actual results if available
    const mockAnalysisResults = { /* Replace with actual analysis results */ };
    visualizeAnalyzerResults(
      'clustering', // Analyzer type
      mockAnalysisResults, // Analysis results data
      visualizationType, // Visualization type based on tab
      { timeFrame: selectedTimeFrame } // Options
    );

    return () => {
      resetAll();
    };
  }, [selectedTimeFrame, activeTab, visualizeAnalyzerResults, resetAll]);

  const handleRefresh = () => {
    let visualizationType = 'cluster_kmeans';
    if (activeTab === 'hierarchical') visualizationType = 'cluster_hierarchical';
    else if (activeTab === 'dbscan') visualizationType = 'cluster_dbscan';
    else if (activeTab === 'evaluation') visualizationType = 'cluster_evaluation';

    const mockAnalysisResults = { /* Replace with actual analysis results */ };
    visualizeAnalyzerResults('clustering', mockAnalysisResults, visualizationType, { timeFrame: selectedTimeFrame });
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
       // Render raw data (e.g., cluster details)
       // return <YourClusterTable data={chartData.chart_data} />;
       return <div className="rounded-md bg-background-main p-4 text-center text-text-muted" style={{ minHeight }}>クラスタリングデータを表示するコンポーネントが必要です。</div>;
    }

    return <div className="rounded-md bg-background-main p-4 text-center text-text-muted" style={{ minHeight }}>表示するデータがありません。</div>;
  };

  return (
    <div className="h-full p-4">
      <div className="mb-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold">クラスタリング</h2>
          <p className="text-sm text-text-secondary">類似した特性を持つ企業のグループを特定します。</p>
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
          <TabsTrigger value="kmeans">K-means</TabsTrigger>
          <TabsTrigger value="hierarchical">階層的クラスタリング</TabsTrigger>
          <TabsTrigger value="dbscan">DBSCAN</TabsTrigger>
          <TabsTrigger value="evaluation">評価指標</TabsTrigger>
        </TabsList>

        <TabsContent value="kmeans" className="mt-0 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>K-meansクラスタリング</CardTitle>
              <CardDescription>データポイントをK個のクラスタに分類</CardDescription>
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
                <CardTitle>クラスタ概要</CardTitle>
                <CardDescription>各クラスタの主要特性</CardDescription>
              </CardHeader>
              <CardContent>
                 // Mock data removed
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>クラスタ間比較</CardTitle>
                <CardDescription>クラスタ間の主要指標比較</CardDescription>
              </CardHeader>
              <CardContent>
                 // Mock data removed
              </CardContent>
            </Card>
          </div>
          */}
        </TabsContent>

        <TabsContent value="hierarchical" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>階層的クラスタリング</CardTitle>
              <CardDescription>データポイント間の階層的関係を分析</CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent('450px')}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="dbscan" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>DBSCAN分析</CardTitle>
              <CardDescription>密度ベースのクラスタリング</CardDescription>
            </CardHeader>
            <CardContent>
              {renderContent('450px')}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="evaluation" className="mt-0">
          <Card>
            <CardHeader>
              <CardTitle>クラスタリング評価</CardTitle>
              <CardDescription>各クラスタリング手法の評価と比較</CardDescription>
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