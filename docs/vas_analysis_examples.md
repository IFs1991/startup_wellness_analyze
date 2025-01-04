# VASスケール分析システム 実装例

## 1. 相関分析の実装例

### 1.1 相関分析クラス
```python
class CorrelationAnalyzer:
    def __init__(self, bq_service: BigQueryService):
        self.bq_service = bq_service

    async def analyze(
        self,
        query: str,
        vas_variables: List[str],
        financial_variables: List[str],
        save_results: bool = True,
        dataset_id: Optional[str] = None,
        table_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        try:
            # データ取得
            data = await self.bq_service.fetch_data(query)

            # データバリデーション
            is_valid, error_message = self._validate_data(
                data, vas_variables, financial_variables
            )
            if not is_valid:
                raise ValueError(error_message)

            # 相関分析の実行
            selected_data = data[vas_variables + financial_variables]
            correlation_matrix = selected_data.corr()

            # 結果の保存
            if save_results and dataset_id and table_id:
                await self.bq_service.save_results(
                    correlation_matrix,
                    dataset_id=dataset_id,
                    table_id=table_id
                )

            # メタデータの作成
            metadata = {
                'row_count': len(data),
                'vas_variables': vas_variables,
                'financial_variables': financial_variables,
                'correlation_pairs': len(vas_variables) * len(financial_variables)
            }

            return correlation_matrix, metadata

        except Exception as e:
            raise RuntimeError(f"相関分析の実行中にエラーが発生しました: {str(e)}")
```

### 1.2 フロントエンド実装例
```typescript
interface CorrelationMatrixProps {
  data: {
    pain_stress: number;
    pain_sleep: number;
    stress_sleep: number;
  };
}

export const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({ data }) => {
  const getCorrelationColor = (value: number): string => {
    // 相関係数に基づいて背景色を計算
    const intensity = Math.abs(value);
    const hue = value > 0 ? 120 : 0; // 正の相関は緑、負の相関は赤
    return `hsla(${hue}, 80%, 50%, ${intensity * 0.5})`;
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full bg-white border">
        <thead>
          <tr>
            <th className="px-4 py-2">変数</th>
            <th className="px-4 py-2">痛み</th>
            <th className="px-4 py-2">ストレス</th>
            <th className="px-4 py-2">睡眠</th>
          </tr>
        </thead>
        <tbody>
          {/* 相関行列の表示 */}
        </tbody>
      </table>
    </div>
  );
};
```

## 2. 時系列分析の実装例

### 2.1 時系列分析クラス
```python
class TimeSeriesAnalyzer:
    def __init__(self, bq_service: BigQueryService):
        self.bq_service = bq_service

    async def analyze_trends(
        self,
        data: pd.DataFrame,
        target_column: str,
        period: int = 30
    ) -> Dict[str, Any]:
        # トレンド分析の実装
        try:
            # 時系列データの準備
            ts_data = data.set_index('timestamp')[target_column]

            # 移動平均の計算
            ma = ts_data.rolling(window=period).mean()

            # トレンドの検出
            decomposition = sm.tsa.seasonal_decompose(
                ts_data,
                period=period,
                extrapolate_trend='freq'
            )

            return {
                'trend': decomposition.trend.tolist(),
                'seasonal': decomposition.seasonal.tolist(),
                'resid': decomposition.resid.tolist(),
                'moving_average': ma.tolist()
            }

        except Exception as e:
            raise RuntimeError(f"トレンド分析中にエラーが発生: {str(e)}")
```

### 2.2 APIエンドポイント実装例
```python
@router.post("/time-series", response_model=AnalysisResponse)
async def analyze_time_series_endpoint(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    analysis_service: AnalysisService = Depends(get_analysis_service)
) -> AnalysisResponse:
    """時系列分析を実行するエンドポイント"""
    try:
        if not request.target_column:
            raise HTTPException(
                status_code=400,
                detail="target_column is required"
            )

        result = await analysis_service.time_series_analysis(
            request.collection_name,
            request.target_column,
            request.conditions,
            request.periods
        )
        return AnalysisResponse(data=result)
    except Exception as e:
        logging.error(f"時系列分析エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

## 3. データバリデーションの実装例

### 3.1 バリデーション関数
```python
def validate_vas_data(
    data: pd.DataFrame,
    required_columns: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    VASデータのバリデーション

    Args:
        data: 検証対象のデータフレーム
        required_columns: 必須カラムのリスト

    Returns:
        (バリデーション結果, エラーメッセージ)
    """
    # データの存在確認
    if data.empty:
        return False, "データが空です"

    # 必須カラムの確認
    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]
    if missing_columns:
        return False, f"必須カラムがありません: {', '.join(missing_columns)}"

    # データ型の確認
    non_numeric = [
        col for col in required_columns
        if not pd.api.types.is_numeric_dtype(data[col])
    ]
    if non_numeric:
        return False, f"数値型ではないカラムがあります: {', '.join(non_numeric)}"

    # VASスケールの範囲確認 (0-10)
    vas_columns = [col for col in required_columns if 'vas' in col.lower()]
    for col in vas_columns:
        if not all(data[col].between(0, 10)):
            return False, f"{col}の値が有効範囲(0-10)外です"

    return True, None
```

### 3.2 エラーハンドリングの実装例
```python
class AnalysisError(Exception):
    """分析処理に関するエラー"""
    pass

class ValidationError(Exception):
    """データバリデーションに関するエラー"""
    pass

async def safe_analyze(
    data: pd.DataFrame,
    config: AnalysisConfig
) -> Dict[str, Any]:
    """
    エラーハンドリング付きの分析実行

    Args:
        data: 分析対象データ
        config: 分析設定

    Returns:
        分析結果
    """
    try:
        # データバリデーション
        is_valid, error_message = validate_vas_data(
            data,
            config.required_columns
        )
        if not is_valid:
            raise ValidationError(error_message)

        # 分析の実行
        result = await perform_analysis(data, config)

        # 結果の検証
        if not validate_results(result):
            raise AnalysisError("分析結果が無効です")

        return result

    except ValidationError as e:
        logger.error(f"バリデーションエラー: {str(e)}")
        raise

    except AnalysisError as e:
        logger.error(f"分析エラー: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"予期せぬエラー: {str(e)}")
        raise
```

## 4. データ永続化の実装例

### 4.1 Firestore保存
```python
class FirestoreService:
    def __init__(self):
        self.db = firestore.Client()

    async def save_analysis_results(
        self,
        results: Dict[str, Any],
        collection: str,
        document_id: Optional[str] = None
    ) -> str:
        """
        分析結果をFirestoreに保存

        Args:
            results: 保存する分析結果
            collection: コレクション名
            document_id: ドキュメントID（省略可）

        Returns:
            保存されたドキュメントのID
        """
        try:
            # タイムスタンプの追加
            results['created_at'] = firestore.SERVER_TIMESTAMP

            # ドキュメントの保存
            if document_id:
                doc_ref = self.db.collection(collection).document(document_id)
                await doc_ref.set(results)
                return document_id
            else:
                doc_ref = self.db.collection(collection).document()
                await doc_ref.set(results)
                return doc_ref.id

        except Exception as e:
            logger.error(f"Firestore保存エラー: {str(e)}")
            raise
```

### 4.2 BigQuery保存
```python
class BigQueryService:
    def __init__(self):
        self.client = bigquery.Client()

    async def save_results(
        self,
        data: pd.DataFrame,
        dataset_id: str,
        table_id: str
    ) -> None:
        """
        分析結果をBigQueryに保存

        Args:
            data: 保存するデータフレーム
            dataset_id: データセットID
            table_id: テーブルID
        """
        try:
            # テーブル参照の作成
            table_ref = self.client.dataset(dataset_id).table(table_id)

            # データの保存
            job_config = bigquery.LoadJobConfig()
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

            job = self.client.load_table_from_dataframe(
                data,
                table_ref,
                job_config=job_config
            )
            job.result()  # 完了を待機

        except Exception as e:
            logger.error(f"BigQuery保存エラー: {str(e)}")
            raise
```

## 5. フロントエンド可視化の実装例

### 5.1 時系列チャートコンポーネント
```typescript
interface TimeSeriesChartProps {
  vasData: {
    timestamp: string;
    value: number;
  }[];
  financialData: {
    timestamp: string;
    value: number;
  }[];
}

export const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  vasData,
  financialData
}) => {
  const chartData = {
    labels: vasData.map(d => d.timestamp),
    datasets: [
      {
        label: 'VASスコア',
        data: vasData.map(d => d.value),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
      {
        label: '財務指標',
        data: financialData.map(d => d.value),
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1
      }
    ]
  };

  return (
    <Line
      data={chartData}
      options={{
        responsive: true,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        scales: {
          y: {
            type: 'linear',
            display: true,
            position: 'left',
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            grid: {
              drawOnChartArea: false,
            },
          },
        }
      }}
    />
  );
};
```