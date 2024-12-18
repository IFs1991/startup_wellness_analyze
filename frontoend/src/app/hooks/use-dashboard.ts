import { useState } from "react"
import { useToast } from "@/components/ui/use-toast"
import {
  DashboardConfig,
  VisualizationResponse,
  ApiResponse,
  api
} from "@/lib/api"  // 既存のapi.tsから型定義とapiオブジェクトをインポート

export function useDashboard() {
  // ローディング状態を管理するためのstate
  const [loading, setLoading] = useState(false)
  // トースト通知のためのフック
  const { toast } = useToast()

  /**
   * ダッシュボードを作成する関数
   * @param config ダッシュボードの設定情報
   * @returns 作成されたダッシュボードの情報
   */
  const createDashboard = async (config: DashboardConfig): Promise<VisualizationResponse> => {
    setLoading(true)
    try {
      // api.createDashboardを使用してダッシュボードを作成
      // このメソッドはApiResponse<VisualizationResponse>を返します
      const response = await api.createDashboard(config)

      // 成功時のトースト通知
      toast({
        title: "ダッシュボードを作成しました",
        description: "新しいダッシュボードが正常に作成されました。",
      })

      // レスポンスからデータを取り出して返す
      return response.data
    } catch (error) {
      // エラー時のトースト通知
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

  /**
   * ユーザーの可視化データを取得する関数
   * @returns ユーザーの可視化データの配列
   */
  const getUserVisualizations = async (): Promise<VisualizationResponse[]> => {
    setLoading(true)
    try {
      // api.getUserVisualizationsを使用してデータを取得
      const response = await api.getUserVisualizations()
      return response.data
    } catch (error) {
      // エラー時のトースト通知
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

  // フックの戻り値として関数とローディング状態を返す
  return {
    loading,
    createDashboard,
    getUserVisualizations,
  }
}