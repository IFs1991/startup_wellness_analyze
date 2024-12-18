// src/types/index.ts

/**
 * ダッシュボード関連の型定義
 */

// ダッシュボードの設定情報を表す型
export interface DashboardConfig {
    title: string                      // ダッシュボードのタイトル
    description?: string               // ダッシュボードの説明（オプション）
    widgets: DashboardWidget[]         // ダッシュボードに含まれるウィジェットの配列
    layout: DashboardLayout           // ウィジェットのレイアウト情報
  }

  // ダッシュボードのレイアウト情報
  export interface DashboardLayout {
    [key: string]: {
      x: number                       // グリッド上のX座標
      y: number                       // グリッド上のY座標
      w: number                       // ウィジェットの幅
      h: number                       // ウィジェットの高さ
      static?: boolean                // 固定レイアウトかどうか
    }
  }

  // ダッシュボードのウィジェット設定
  export interface DashboardWidget {
    id: string                        // ウィジェットの一意識別子
    type: WidgetType                  // ウィジェットの種類
    title: string                     // ウィジェットのタイトル
    settings?: Record<string, any>    // ウィジェット固有の設定
    dataSource?: string               // データソースの指定
  }

  // ウィジェットの種類を定義
  export type WidgetType = 'graph' | 'table' | 'metric' | 'text'

  /**
   * グラフ関連の型定義
   */

  // グラフの設定情報
  export interface GraphConfig {
    type: GraphType                   // グラフの種類
    title: string                     // グラフのタイトル
    data_source: string              // データソースのパス
    settings: GraphSettings          // グラフの詳細設定
    filters?: DataFilter[]           // データフィルターの配列
  }

  // グラフの種類を定義
  export type GraphType = 'line' | 'bar' | 'pie' | 'scatter' | 'area'

  // グラフの詳細設定
  export interface GraphSettings {
    xAxis?: {
      field: string                  // X軸のデータフィールド
      label?: string                 // X軸のラベル
    }
    yAxis?: {
      field: string                  // Y軸のデータフィールド
      label?: string                 // Y軸のラベル
    }
    colors?: string[]               // グラフの色設定
    legend?: boolean               // 凡例の表示設定
    [key: string]: any             // その他のカスタム設定
  }

  /**
   * データフィルター関連の型定義
   */

  // データフィルターの設定
  export interface DataFilter {
    field: string                   // フィルター対象のフィールド
    operator: FilterOperator        // フィルター演算子
    value: any                      // フィルター値
  }

  // フィルター演算子の種類
  export type FilterOperator =
    | 'equals'
    | 'notEquals'
    | 'greaterThan'
    | 'lessThan'
    | 'contains'
    | 'between'

  /**
   * API関連の型定義
   */

  // APIレスポンスの基本型
  export interface ApiResponse<T> {
    data: T                         // レスポンスデータ
    status: number                  // ステータスコード
    message?: string                // メッセージ（オプション）
  }

  // 可視化データのレスポンス型
  export interface VisualizationResponse {
    id: string                      // 可視化データのID
    created_at: string              // 作成日時
    updated_at: string              // 更新日時
    config: Record<string, any>     // 設定情報
    data: Record<string, any>       // 実際のデータ
    created_by: string              // 作成者のID
  }

  /**
   * ユーティリティ型
   */

  // ページネーション用のパラメータ
  export interface PaginationParams {
    page: number                    // ページ番号
    limit: number                   // 1ページあたりの件数
  }

  // ソート用のパラメータ
  export interface SortParams {
    field: string                   // ソート対象のフィールド
    order: 'asc' | 'desc'          // ソート順序
  }