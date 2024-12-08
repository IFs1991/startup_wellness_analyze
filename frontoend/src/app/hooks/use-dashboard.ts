import { useState } from "react"
import { useToast } from "@/components/ui/use-toast"
import { DashboardConfig, VisualizationResponse } from "@/types"
import { visualizationApi } from "@/lib/api"

export function useDashboard() {
  const [loading, setLoading] = useState(false)
  const { toast } = useToast()

  const createDashboard = async (config: DashboardConfig) => {
    setLoading(true)
    try {
      const response = await visualizationApi.createDashboard(config)
      toast({
        title: "ダッシュボードを作成しました",
        description: "新しいダッシュボードが正常に作成されました。",
      })
      return response
    } catch (error) {
      toast({
        title: "エラー",
        description: "ダッシュボードの作成に失敗しました。",
        variant: "destructive",
      })
      throw error
    } finally {
      setLoading(false)
    }
  }

  const getUserVisualizations = async () => {
    setLoading(true)
    try {
      const visualizations = await visualizationApi.getUserVisualizations()
      return visualizations
    } catch (error) {
      toast({
        title: "エラー",
        description: "可視化データの取得に失敗しました。",
        variant: "destructive",
      })
      throw error
    } finally {
      setLoading(false)
    }
  }

  return {
    loading,
    createDashboard,
    getUserVisualizations,
  }