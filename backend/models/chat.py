import os
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
from google.cloud import aiplatform
from google.oauth2 import service_account
import logging
import uuid
import json
# SQLAlchemyのインポート
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import Session, relationship
from database.models import Base, User, Company
from database.connection import get_db
from datetime import datetime
# 分析モジュールのインポート
from backend.analysis import AnalysisType, AnalysisRegistry, AnalysisResult
# AI分析関連のインポート
from backend.models.ai_analysis import (
    AIAnalyzer, CompanyAnalysisContext, AIAnalysisResponse,
    DataPreprocessor, AIAnalysisRequest
)

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# アプリケーション初期化
app = FastAPI(title="Vertex AI Agent Builder API")

# 認証設定
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# デフォルトの前処理クラス
class DefaultDataPreprocessor(DataPreprocessor):
    """デフォルトのデータ前処理クラス"""
    async def preprocess(self, data: Any) -> Any:
        """
        データの前処理を行う
        Args:
            data: 前処理対象のデータ
        Returns:
            前処理済みのデータ
        """
        logger.info("データ前処理の実行")
        return data

# モデル定義
class Message(BaseModel):
    """会話メッセージモデル"""
    role: str = Field(..., description="メッセージの役割（'user'または'assistant'）")
    content: str = Field(..., description="メッセージ内容")

class AnalysisRequest(BaseModel):
    """分析リクエストモデル"""
    analysis_type: AnalysisType = Field(..., description="実行する分析のタイプ")
    params: Dict = Field(default_factory=dict, description="分析パラメータ")

class ChatRequest(BaseModel):
    """チャットリクエストモデル"""
    message: str = Field(..., description="ユーザーからのメッセージ")
    company_id: str = Field(..., description="会社ID")
    agent_id: str = Field("", description="使用するVertex AI Agentのエージェント ID")
    location: str = Field("us-central1", description="エージェント所在地")
    project_id: str = Field("", description="GCPプロジェクトID")
    session_id: Optional[str] = Field(None, description="セッションID（新規セッションの場合は空）")
    user_id: str = Field(..., description="ユーザーID")
    analysis_request: Optional[AnalysisRequest] = Field(None, description="分析リクエスト（オプション）")
    enable_company_context: bool = Field(True, description="企業コンテキストを有効化するかどうか")

class ChatResponse(BaseModel):
    """拡張されたチャットレスポンスモデル"""
    response: str = Field(..., description="アシスタントからの応答")
    session_id: str = Field(..., description="会話セッションID")
    request_id: str = Field(..., description="リクエストID")
    analysis_result: Optional[Dict] = Field(None, description="分析結果（存在する場合）")
    visualizations: Optional[List[Dict[str, Any]]] = Field(None, description="可視化要素")
    interactive_elements: Optional[List[Dict[str, Any]]] = Field(None, description="インタラクティブ要素")
    suggested_actions: Optional[List[Dict[str, Any]]] = Field(None, description="推奨アクション")
    company_insights: Optional[List[Dict[str, Any]]] = Field(None, description="企業インサイト")

class ChatSession(Base):
    """チャットセッションモデル"""
    __tablename__ = 'chat_sessions'

    id = Column(String, primary_key=True)
    company_id = Column(String, ForeignKey('companies.id'), nullable=False)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    agent_session_id = Column(String, nullable=True)  # Vertex AI Agent Builder のセッションID
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    context_data = Column(JSON, nullable=True)  # 企業コンテキストデータ

    # リレーションシップ
    company = relationship("Company", back_populates="chat_sessions")
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    """チャットメッセージモデル"""
    __tablename__ = 'chat_messages'

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('chat_sessions.id'), nullable=False)
    role = Column(String, nullable=False)  # user or assistant
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    analysis_result = Column(JSON, nullable=True)  # 分析結果を保存

    # リレーションシップ
    session = relationship("ChatSession", back_populates="messages")

# エージェントクライアントキャッシュ
agent_client_cache = {}

# AIアナライザーインスタンス
ai_analyzer = AIAnalyzer(DefaultDataPreprocessor())

# 企業データ境界クラス - 企業データの分離を確保
class EnterpriseDataBoundary:
    """企業データの境界を明示的に管理するクラス"""

    def __init__(self, company_id: str):
        self.company_id = company_id

    async def get_analysis_context(self, db: Session) -> CompanyAnalysisContext:
        """
        企業の分析コンテキストを取得
        Args:
            db: データベースセッション
        Returns:
            企業分析コンテキスト
        """
        # 企業情報の取得
        company = db.query(Company).filter(Company.id == self.company_id).first()
        if not company:
            logger.error(f"企業が見つかりません: {self.company_id}")
            raise ValueError(f"企業が見つかりません: {self.company_id}")

        # 財務データ、ウェルネスデータなどの取得（実際の実装ではデータベースから取得）
        # サンプル実装
        return CompanyAnalysisContext(
            company=company,
            financial_data=[],  # 実際の実装ではデータベースから取得
            financial_ratios=[],  # 実際の実装ではデータベースから取得
            financial_growth=[],  # 実際の実装ではデータベースから取得
            wellness_metrics=[],  # 実際の実装ではデータベースから取得
            wellness_trends=[],  # 実際の実装ではデータベースから取得
        )

    async def get_recent_analyses(self, db: Session, limit: int = 5) -> List[AIAnalysisResponse]:
        """
        最近の分析結果を取得
        Args:
            db: データベースセッション
            limit: 取得する分析結果の最大数
        Returns:
            分析結果のリスト
        """
        # 実際の実装ではデータベースから取得
        # ここではサンプル実装
        return []

    async def log_data_access(self, user_id: str, access_type: str, data_accessed: List[str], db: Session):
        """
        データアクセスの詳細なログを記録
        Args:
            user_id: ユーザーID
            access_type: アクセスタイプ
            data_accessed: アクセスされたデータの種類
            db: データベースセッション
        """
        # アクセスログをデータベースに記録
        # 実際の実装ではデータベースに保存
        logger.info(f"データアクセスログ: 企業={self.company_id}, ユーザー={user_id}, タイプ={access_type}, データ={data_accessed}")

# Vertex AI Agent Builder クライアント初期化関数
def init_agent_client(
    project_id: str,
    location: str = "us-central1",
    credentials_path: Optional[str] = None
):
    """
    Vertex AI Agent Builder クライアントを初期化
    Args:
        project_id: GCPプロジェクトID
        location: リージョン
        credentials_path: サービスアカウントキーパス（オプション）
    Returns:
        初期化された Vertex AI Agent クライアント
    """
    try:
        # キャッシュをチェック
        cache_key = f"{project_id}:{location}"
        if cache_key in agent_client_cache:
            logger.info(f"キャッシュからエージェントクライアントを使用: {cache_key}")
            return agent_client_cache[cache_key]

        # 認証情報の設定
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            client_options = {"credentials": credentials}
        else:
            client_options = {}

        # Agent Builder クライアントの初期化
        agent_client = aiplatform.VertexAI(
            project=project_id,
            location=location,
            **client_options
        ).get_vertex_ai_agent_client()

        # キャッシュに保存
        agent_client_cache[cache_key] = agent_client

        return agent_client
    except Exception as e:
        logger.error(f"Agent Builder クライアント初期化エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent Builder クライアント初期化エラー: {str(e)}"
        )

# 認証検証関数
async def verify_api_key(api_key: str = Depends(oauth2_scheme)):
    """
    APIキーを検証
    実際の実装ではデータベースなどと照合する必要があります
    Args:
        api_key: 認証用APIキー
    Returns:
        検証済みのAPIキー
    """
    # 実際の実装ではデータベースなどでAPIキーを検証
    if not api_key or len(api_key) < 10:  # 簡易的な検証
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無効なAPIキー"
        )
    return api_key

async def process_analysis_request(
    analysis_request: AnalysisRequest,
    company_id: str,
    db: Session
) -> Optional[AnalysisResult]:
    """
    分析リクエストを処理する
    Args:
        analysis_request: 分析リクエスト
        company_id: 会社ID
        db: データベースセッション
    Returns:
        分析結果（オプション）
    """
    if not analysis_request:
        return None

    try:
        # 企業データ境界の取得
        boundary = EnterpriseDataBoundary(company_id)

        # 分析サービスの取得
        service = AnalysisRegistry.get_service(analysis_request.analysis_type)
        if not service:
            raise ValueError(f"未サポートの分析タイプ: {analysis_request.analysis_type}")

        # 企業分析コンテキストの取得
        analysis_context = await boundary.get_analysis_context(db)

        # データの取得
        data = await get_company_data(company_id, db)

        # 分析の実行
        result = service.analyze(data, analysis_request.params)

        # 分析結果の説明を生成
        explanation = service.explain_result(result)

        # 結果を拡張
        result.metadata["explanation"] = explanation

        return result
    except Exception as e:
        logger.error(f"分析処理エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析処理エラー: {str(e)}"
        )

async def get_company_data(company_id: str, db: Session) -> Any:
    """
    会社のデータを取得する
    Args:
        company_id: 会社ID
        db: データベースセッション
    Returns:
        会社データ
    """
    try:
        # 会社情報の取得
        company = db.query(Company).filter(Company.id == company_id).first()
        if not company:
            raise ValueError(f"会社が見つかりません: {company_id}")

        # 実際の実装ではここでVASデータと財務データを取得
        # 例えば:
        # vas_data = db.query(VASData).filter(VASData.company_id == company_id).all()
        # financial_data = db.query(FinancialData).filter(FinancialData.company_id == company_id).all()

        # ここではダミーデータを返す
        return {
            "company_id": company_id,
            "company_name": company.name if company else "Unknown Company",
            "dummy_data": True
        }
    except Exception as e:
        logger.error(f"会社データ取得エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"会社データ取得エラー: {str(e)}"
        )

async def process_chat_session(request: ChatRequest, db: Session) -> ChatSession:
    """
    チャットセッションを処理する関数
    新規セッションの場合は作成し、既存セッションの場合は取得する
    Args:
        request: チャットリクエスト
        db: データベースセッション
    Returns:
        処理済みのチャットセッション
    """
    if request.session_id:
        # 既存セッションの取得
        session = db.query(ChatSession).filter(
            ChatSession.id == request.session_id,
            ChatSession.is_active == True
        ).first()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="無効なセッションID"
            )
    else:
        # 新規セッション作成
        session = ChatSession(
            id=str(uuid.uuid4()),
            company_id=request.company_id,
            user_id=request.user_id,
            agent_session_id=None,  # 後でVertex AI Agent Builder用のセッションIDを設定
            context_data={}  # 企業コンテキストデータの初期化
        )
        db.add(session)
        db.flush()  # IDを生成するためにフラッシュ

    # ユーザーメッセージを追加
    new_message = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session.id,
        role="user",
        content=request.message
    )
    db.add(new_message)
    db.flush()

    return session

async def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    analysis_result: Optional[Dict] = None,
    db: Session = None
) -> ChatMessage:
    """
    チャットメッセージをデータベースに保存する関数
    Args:
        session_id: セッションID
        role: メッセージの役割（'user'または'assistant'）
        content: メッセージ内容
        analysis_result: 分析結果（オプション）
        db: データベースセッション
    Returns:
        保存されたチャットメッセージ
    """
    message = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        content=content,
        analysis_result=analysis_result
    )

    db.add(message)
    db.flush()

    return message

# 企業分析コンテキストを準備する関数
async def prepare_company_context(
    company_id: str,
    db: Session,
    enable_context: bool = True
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    企業の分析コンテキストを準備する
    Args:
        company_id: 企業ID
        db: データベースセッション
        enable_context: コンテキストを有効にするかどうか
    Returns:
        企業コンテキスト情報と企業インサイトのタプル
    """
    if not enable_context:
        return {}, []

    try:
        # 企業データ境界の取得
        boundary = EnterpriseDataBoundary(company_id)

        # 企業分析コンテキストの取得
        analysis_context = await boundary.get_analysis_context(db)

        # 最近の分析結果を取得
        recent_analyses = await boundary.get_recent_analyses(db)

        # 企業インサイトの準備
        company_insights = []
        for analysis in recent_analyses:
            # 分析タイプに基づいて重要度を設定
            importance = "high" if analysis.analysis_type in ["correlation", "regression"] else "medium"

            for insight in analysis.insights:
                company_insights.append({
                    "type": insight.get("type", "insight"),
                    "content": insight.get("content", ""),
                    "importance": importance,
                    "analysis_type": analysis.analysis_type,
                    "date": analysis.analysis_date.isoformat()
                })

        # エージェントに送信するコンテキストを準備
        context_data = {
            "company_name": analysis_context.company.name,
            "industry": analysis_context.company.industry if hasattr(analysis_context.company, "industry") else "Unknown",
            "key_metrics": {
                "wellness_score": "データから抽出", # 実際の実装ではデータから抽出
                "financial_health": "データから抽出", # 実際の実装ではデータから抽出
                "employee_engagement": "データから抽出" # 実際の実装ではデータから抽出
            },
            "recent_trends": [
                {
                    "metric": "wellness_score",
                    "trend": "上昇", # 実際の実装ではデータから抽出
                    "period": "過去3ヶ月"
                }
            ]
        }

        # データアクセスのログを記録
        await boundary.log_data_access(
            "system",  # システムアクセス
            "context_preparation",
            ["company_info", "wellness_metrics", "financial_data"],
            db
        )

        return context_data, company_insights

    except Exception as e:
        logger.error(f"企業コンテキスト準備エラー: {str(e)}")
        return {}, []

# エージェントと会話を行う関数
async def converse_with_agent(
    agent_client,
    agent_id: str,
    session_id: Optional[str],
    message: str,
    company_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Vertex AI Agent Builderエージェントと会話する
    Args:
        agent_client: エージェントクライアント
        agent_id: エージェントID
        session_id: エージェントセッションID（新規の場合はNone）
        message: ユーザーメッセージ
        company_context: 企業コンテキスト情報（オプション）
    Returns:
        レスポンス情報を含む辞書
    """
    try:
        # セッションの作成または取得
        if not session_id:
            session = agent_client.create_session(agent=agent_id)
            session_id = session.name.split('/')[-1]
            logger.info(f"新規エージェントセッション作成: {session_id}")

            # 新規セッションの場合、企業コンテキストを初期メッセージとして送信
            if company_context:
                context_message = f"システム情報: 以下は分析対象の企業に関する重要な情報です。\n{json.dumps(company_context, ensure_ascii=False, indent=2)}"

                # コンテキスト情報を送信
                agent_client.converse_session(
                    session=f"projects/{agent_client.project}/locations/{agent_client.location}/agents/{agent_id}/sessions/{session_id}",
                    message=context_message
                )
                logger.info("企業コンテキスト情報をエージェントに送信しました")
        else:
            logger.info(f"既存エージェントセッション使用: {session_id}")

        # メッセージ送信と応答取得
        response = agent_client.converse_session(
            session=f"projects/{agent_client.project}/locations/{agent_client.location}/agents/{agent_id}/sessions/{session_id}",
            message=message
        )

        # 応答の処理
        assistant_response = ""
        for message in response.conversation.messages:
            if message.author == "agent":
                assistant_response += message.content

        # 追加情報の抽出（ツール呼び出し結果、関数呼び出し結果など）
        tool_outputs = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_outputs.append({
                    "tool": tool_call.tool,
                    "output": tool_call.output if hasattr(tool_call, 'output') else None
                })

        return {
            "session_id": session_id,
            "response": assistant_response,
            "tool_outputs": tool_outputs
        }
    except Exception as e:
        logger.error(f"エージェント会話エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"エージェント会話エラー: {str(e)}"
        )

# 分析結果を処理する関数
async def process_analysis_results(analysis_result: Optional[AnalysisResult]) -> Dict[str, Any]:
    """
    分析結果を処理し、UI表示用のデータを準備する
    Args:
        analysis_result: 分析結果
    Returns:
        UI表示用データを含む辞書
    """
    if not analysis_result:
        return {
            "visualizations": None,
            "interactive_elements": None,
            "suggested_actions": None
        }

    try:
        # 可視化データの準備
        visualizations = []
        if "graph_data" in analysis_result.metadata:
            visualizations.append({
                "type": "graph",
                "data": analysis_result.metadata["graph_data"]
            })
        if "table_data" in analysis_result.metadata:
            visualizations.append({
                "type": "table",
                "data": analysis_result.metadata["table_data"]
            })

        # インタラクティブ要素の準備
        interactive_elements = []
        if "filters" in analysis_result.metadata:
            for filter_item in analysis_result.metadata["filters"]:
                interactive_elements.append({
                    "type": "filter",
                    "options": filter_item["options"],
                    "default": filter_item.get("default", ""),
                    "label": filter_item.get("label", "フィルター")
                })

        # 推奨アクションの準備
        suggested_actions = analysis_result.metadata.get("recommendations", None)

        return {
            "visualizations": visualizations if visualizations else None,
            "interactive_elements": interactive_elements if interactive_elements else None,
            "suggested_actions": suggested_actions
        }
    except Exception as e:
        logger.error(f"分析結果処理エラー: {str(e)}")
        return {
            "visualizations": None,
            "interactive_elements": None,
            "suggested_actions": None
        }

# メッセージから分析ニーズを検出する関数
async def detect_analysis_needs(message: str) -> Optional[AIAnalysisRequest]:
    """
    ユーザーメッセージから分析ニーズを検出する
    Args:
        message: ユーザーメッセージ
    Returns:
        検出された分析リクエスト（オプション）
    """
    # 実際の実装では、NLPを使用してメッセージから分析ニーズを検出
    # ここではシンプルなキーワードマッチングの例
    analysis_keywords = {
        "相関": AnalysisType.CORRELATION,
        "トレンド": AnalysisType.TIME_SERIES,
        "予測": AnalysisType.REGRESSION,
        "グループ": AnalysisType.CLUSTER,
        "生存": AnalysisType.SURVIVAL,
        "テキスト": AnalysisType.TEXT_MINING,
        "確率": AnalysisType.BAYESIAN,
        "主成分": AnalysisType.PCA,
        "関連": AnalysisType.ASSOCIATION
    }

    # 簡易的な分析タイプの検出
    for keyword, analysis_type in analysis_keywords.items():
        if keyword in message:
            return AIAnalysisRequest(
                company_id="",  # 後で設定
                analysis_type=analysis_type.value,
                metrics=None,  # 実際の実装ではメッセージから抽出
                time_range=None  # 実際の実装ではメッセージから抽出
            )

    return None

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Vertex AI Agent Builder を使用したチャットエンドポイント（AI分析統合版）
    """
    request_id = str(uuid.uuid4())
    logger.info(f"リクエスト受信: {request_id}")

    try:
        # エージェントクライアントの初期化
        agent_client = init_agent_client(
            project_id=request.project_id,
            location=request.location
        )

        # チャットセッションの処理
        session = await process_chat_session(request, db)

        # 企業コンテキストの準備
        company_context, company_insights = await prepare_company_context(
            request.company_id,
            db,
            request.enable_company_context
        )

        # セッションにコンテキストデータを保存
        if company_context and not session.context_data:
            session.context_data = company_context
            db.add(session)
            db.flush()

        # 分析リクエストの処理
        analysis_result = None
        ai_analysis_response = None

        if request.analysis_request:
            # 明示的な分析リクエストの処理
            analysis_result = await process_analysis_request(
                request.analysis_request,
                request.company_id,
                db
            )
        else:
            # メッセージから暗黙的な分析ニーズを検出
            analysis_request = await detect_analysis_needs(request.message)
            if analysis_request:
                # 検出された分析ニーズに応じて分析を実行
                analysis_request.company_id = request.company_id

                # 企業データ境界の取得
                boundary = EnterpriseDataBoundary(request.company_id)

                # 企業分析コンテキストの取得
                analysis_context = await boundary.get_analysis_context(db)

                # AIアナライザーによる分析実行
                ai_analysis_response = await ai_analyzer.analyze_company(
                    analysis_context,
                    AnalysisType(analysis_request.analysis_type),
                    analysis_request.additional_context or {}
                )

        # エージェントとの会話
        agent_response = await converse_with_agent(
            agent_client,
            request.agent_id,
            session.agent_session_id,
            request.message,
            company_context if not session.agent_session_id else None  # 初回のみコンテキスト送信
        )

        # セッションにエージェントセッションIDを保存
        if not session.agent_session_id:
            session.agent_session_id = agent_response["session_id"]
            db.add(session)
            db.flush()

        # アシスタントメッセージを保存
        await save_chat_message(
            session.id,
            "assistant",
            agent_response["response"],
            analysis_result.to_dict() if analysis_result else (
                ai_analysis_response.dict() if ai_analysis_response else None
            ),
            db
        )

        # 分析結果の処理
        ui_data = await process_analysis_results(analysis_result)

        # トランザクションのコミット
        db.commit()

        # 拡張されたレスポンスの作成
        return ChatResponse(
            response=agent_response["response"],
            session_id=session.id,
            request_id=request_id,
            analysis_result=analysis_result.to_dict() if analysis_result else (
                ai_analysis_response.dict() if ai_analysis_response else None
            ),
            visualizations=ui_data["visualizations"],
            interactive_elements=ui_data["interactive_elements"],
            suggested_actions=ui_data["suggested_actions"],
            company_insights=company_insights if company_insights else None
        )
    except Exception as e:
        db.rollback()
        logger.error(f"チャット処理エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"チャット処理エラー: {str(e)}"
        )

# セッションクリアエンドポイント
@app.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    指定されたセッションを無効化
    Args:
        session_id: 無効化するセッションID
        api_key: 認証用APIキー
        db: データベースセッション
    Returns:
        無効化結果
    """
    try:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.is_active == True
        ).first()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="セッションが見つかりません"
            )

        # セッションを無効化
        session.is_active = False
        db.commit()

        return {"status": "success", "message": "セッションが無効化されました"}
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"セッション無効化エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"セッション無効化エラー: {str(e)}"
        )

# メイン実行関数
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)