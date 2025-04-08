"use client";

import React, { useState } from "react";
import { AlertCircle, BarChart3, ChevronDown, Gauge, LineChart, NetworkIcon, PieChart, Tally3, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { analysisExplanations } from "@/lib/analysis-explanations";

// TypeScript型エラーを修正するための型アサーション
const analysisExplanationsRecord = analysisExplanations as Record<string, {
  title: string;
  description: string;
  businessValue: string;
  caution: string;
}>;

// 分析タイプごとのアイコンを定義
const ANALYSIS_ICONS: Record<string, React.ReactNode> = {
  correlation: <NetworkIcon className="h-4 w-4" />,
  timeseries: <LineChart className="h-4 w-4" />,
  cluster: <PieChart className="h-4 w-4" />,
  bayesian: <Gauge className="h-4 w-4" />,
  regression: <LineChart className="h-4 w-4" />,
  wellness: <Tally3 className="h-4 w-4" />,
  financial: <BarChart3 className="h-4 w-4" />,
  default: <Wand2 className="h-4 w-4" />
};

// パラメータエディタの型定義
interface ParameterEditorProps {
  parameters: Record<string, any>;
  onChange: (key: string, value: any) => void;
  analysisType: string;
}

// 分析リクエストボタンの型定義
interface AnalysisRequestButtonProps {
  onRequestAnalysis: (analysisType: string, parameters?: Record<string, any>) => void;
  className?: string;
  disabled?: boolean;
}

// パラメータエディタコンポーネント
function ParameterEditor({ parameters, onChange, analysisType }: ParameterEditorProps) {
  // 分析タイプによって異なるパラメータUIを表示
  switch (analysisType) {
    case "timeseries":
      return (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="timeRange">データ範囲</Label>
            <div className="flex space-x-2">
              <select
                id="timeRange"
                value={parameters.timeRange || "6months"}
                onChange={(e) => onChange("timeRange", e.target.value)}
                className="rounded-md border border-input px-3 py-1 text-sm"
              >
                <option value="3months">3ヶ月</option>
                <option value="6months">6ヶ月</option>
                <option value="1year">1年</option>
                <option value="2years">2年</option>
                <option value="all">全期間</option>
              </select>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="granularity">データ粒度</Label>
            <div className="flex space-x-2">
              <select
                id="granularity"
                value={parameters.granularity || "monthly"}
                onChange={(e) => onChange("granularity", e.target.value)}
                className="rounded-md border border-input px-3 py-1 text-sm"
              >
                <option value="daily">日次</option>
                <option value="weekly">週次</option>
                <option value="monthly">月次</option>
                <option value="quarterly">四半期</option>
              </select>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="smoothing" className="mb-1 block">傾向曲線を表示</Label>
            </div>
            <Switch
              id="smoothing"
              checked={parameters.smoothing || false}
              onCheckedChange={(checked) => onChange("smoothing", checked)}
            />
          </div>
        </div>
      );

    case "correlation":
      return (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="minCorrelation">最小相関値</Label>
            <div className="flex space-x-2 items-center">
              <Input
                id="minCorrelation"
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={parameters.minCorrelation || 0.3}
                onChange={(e) => onChange("minCorrelation", parseFloat(e.target.value))}
                className="w-20 h-8"
              />
            </div>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="showNegative" className="mb-1 block">負の相関を表示</Label>
            </div>
            <Switch
              id="showNegative"
              checked={parameters.showNegative || true}
              onCheckedChange={(checked) => onChange("showNegative", checked)}
            />
          </div>
        </div>
      );

    case "cluster":
      return (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="clusterCount">クラスター数</Label>
            <div className="flex space-x-2 items-center">
              <Input
                id="clusterCount"
                type="number"
                min="2"
                max="10"
                value={parameters.clusterCount || 4}
                onChange={(e) => onChange("clusterCount", parseInt(e.target.value))}
                className="w-20 h-8"
              />
            </div>
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="algorithm">アルゴリズム</Label>
            <div className="flex space-x-2">
              <select
                id="algorithm"
                value={parameters.algorithm || "kmeans"}
                onChange={(e) => onChange("algorithm", e.target.value)}
                className="rounded-md border border-input px-3 py-1 text-sm"
              >
                <option value="kmeans">K-means</option>
                <option value="hierarchical">階層的クラスタリング</option>
                <option value="dbscan">DBSCAN</option>
              </select>
            </div>
          </div>
        </div>
      );

    // デフォルトは空のパラメータエディタ
    default:
      if (Object.keys(parameters).length === 0) {
        return (
          <div className="flex items-center justify-center py-4 text-sm text-muted-foreground">
            <AlertCircle className="h-4 w-4 mr-2" />
            この分析タイプにはパラメータがありません
          </div>
        );
      }

      return (
        <div className="space-y-4">
          {Object.entries(parameters).map(([key, value]) => (
            <div key={key} className="flex items-center justify-between">
              <Label htmlFor={key}>{key}</Label>
              {typeof value === "boolean" ? (
                <Switch
                  id={key}
                  checked={value}
                  onCheckedChange={(checked) => onChange(key, checked)}
                />
              ) : typeof value === "number" ? (
                <Input
                  id={key}
                  type="number"
                  value={value}
                  onChange={(e) => onChange(key, parseFloat(e.target.value))}
                  className="w-20 h-8"
                />
              ) : (
                <Input
                  id={key}
                  value={value as string}
                  onChange={(e) => onChange(key, e.target.value)}
                  className="w-40 h-8"
                />
              )}
            </div>
          ))}
        </div>
      );
  }
}

export function AnalysisRequestButton({
  onRequestAnalysis,
  className,
  disabled = false
}: AnalysisRequestButtonProps) {
  const [selectedAnalysisType, setSelectedAnalysisType] = useState<string | null>(null);
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const [showDialog, setShowDialog] = useState(false);

  // 分析タイプのリスト（利用可能な分析タイプ）
  const analysisTypes = Object.keys(analysisExplanations);

  // パラメータを更新する関数
  const handleParameterChange = (key: string, value: any) => {
    setParameters((prev) => ({
      ...prev,
      [key]: value
    }));
  };

  // 分析タイプを選択する関数
  const handleSelectAnalysisType = (analysisType: string) => {
    setSelectedAnalysisType(analysisType);

    // 分析タイプによって初期パラメータを設定
    switch (analysisType) {
      case "timeseries":
        setParameters({
          timeRange: "6months",
          granularity: "monthly",
          smoothing: false
        });
        break;
      case "correlation":
        setParameters({
          minCorrelation: 0.3,
          showNegative: true
        });
        break;
      case "cluster":
        setParameters({
          clusterCount: 4,
          algorithm: "kmeans"
        });
        break;
      default:
        setParameters({});
    }

    setShowDialog(true);
  };

  // 分析リクエストを確定する関数
  const handleConfirmAnalysisRequest = () => {
    if (selectedAnalysisType) {
      onRequestAnalysis(selectedAnalysisType, parameters);
      setShowDialog(false);
    }
  };

  return (
    <>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className={cn(
                    "h-8 gap-1 px-2 text-xs",
                    className
                  )}
                  disabled={disabled}
                >
                  <BarChart3 className="h-3.5 w-3.5" />
                  分析実行
                  <ChevronDown className="h-3.5 w-3.5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuLabel>分析タイプを選択</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {analysisTypes.map((type) => (
                  <DropdownMenuItem
                    key={type}
                    onClick={() => handleSelectAnalysisType(type)}
                    className="flex items-center gap-2"
                  >
                    {ANALYSIS_ICONS[type] || ANALYSIS_ICONS.default}
                    <span className="capitalize">{analysisExplanationsRecord[type]?.title || type}</span>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </TooltipTrigger>
          <TooltipContent side="left">
            <p>データ分析を実行</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {/* 分析パラメータ設定ダイアログ */}
      <Dialog open={showDialog} onOpenChange={setShowDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {selectedAnalysisType && (
                <div className="flex items-center gap-2">
                  {ANALYSIS_ICONS[selectedAnalysisType] || ANALYSIS_ICONS.default}
                  <span>
                    {analysisExplanationsRecord[selectedAnalysisType]?.title || selectedAnalysisType}分析
                  </span>
                </div>
              )}
            </DialogTitle>
            <DialogDescription>
              {selectedAnalysisType && analysisExplanationsRecord[selectedAnalysisType]?.description}
            </DialogDescription>
          </DialogHeader>

          {/* パラメータエディタ */}
          {selectedAnalysisType && (
            <div className="py-4">
              <ParameterEditor
                parameters={parameters}
                onChange={handleParameterChange}
                analysisType={selectedAnalysisType}
              />
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDialog(false)}>キャンセル</Button>
            <Button onClick={handleConfirmAnalysisRequest}>分析実行</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}