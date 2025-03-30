"""
Neo4jデータベース接続と操作を管理するモジュール

このモジュールはNeo4jグラフデータベースとの接続、セッション管理、
および基本的な操作インターフェースを提供します。
"""

import os
from typing import (
    Optional, Dict, Any, Generator as PyGenerator, List,
    TypeVar, Union, cast, Type as PyType
)
from contextlib import contextmanager

from neo4j import GraphDatabase


T = TypeVar('T', bound='Neo4jDatabaseManager')


class Neo4jDatabaseManager:
    """
    Neo4jデータベースへの接続と基本操作を管理するクラス
    """

    _instance = None
    _driver = None  # Optional['Driver']

    def __new__(cls: PyType[T], *args, **kwargs) -> T:
        """シングルトンパターンによるインスタンス管理"""
        if cls._instance is None:
            cls._instance = super(Neo4jDatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        """
        Neo4jデータベースマネージャーを初期化

        Args:
            uri: Neo4jデータベースのURI
            username: 接続ユーザー名
            password: 接続パスワード
        """
        if self._driver is None:
            uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
            password = password or os.environ.get("NEO4J_PASSWORD", "password")

            self._driver = GraphDatabase.driver(uri, auth=(username, password))

    @property
    def driver(self):  # -> 'Driver'
        """Neo4jドライバーインスタンスを取得"""
        if self._driver is None:
            raise ValueError("Driver is not initialized")
        return self._driver

    def close(self) -> None:
        """データベース接続を閉じる"""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    @contextmanager
    def session(self):  # -> PyGenerator['Session', None, None]:
        """
        Neo4jセッションのコンテキストマネージャー

        使用例:
            with manager.session() as session:
                result = session.run("MATCH (n) RETURN n LIMIT 5")
        """
        if self._driver is None:
            raise ValueError("Driver is not initialized")
        session = self._driver.session()
        try:
            yield session
        finally:
            session.close()

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None):  # -> 'Result':
        """
        Cypherクエリを実行し結果を返す

        Args:
            query: 実行するCypherクエリ
            params: クエリパラメータ

        Returns:
            クエリ実行結果
        """
        with self.session() as session:
            return session.run(query, params or {})


class Neo4jService:
    """
    Neo4jデータベースへのアクセスを提供するサービスクラス
    アプリケーション固有のグラフデータベース操作を実装
    """

    def __init__(self, db_manager: Optional[Neo4jDatabaseManager] = None):
        """
        Neo4jサービスを初期化

        Args:
            db_manager: Neo4jデータベースマネージャー
        """
        self.db_manager = db_manager or Neo4jDatabaseManager()

    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        新しいノードを作成

        Args:
            label: ノードのラベル
            properties: ノードのプロパティ

        Returns:
            作成されたノードの情報
        """
        query = f"CREATE (n:{label} $props) RETURN n"
        result = self.db_manager.execute_query(query, {"props": properties})
        return result.single()[0].as_dict()

    def find_nodes(self, label: str, conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        条件に一致するノードを検索

        Args:
            label: 検索対象ノードのラベル
            conditions: 検索条件

        Returns:
            ノードのリスト
        """
        where_clause = ""
        if conditions:
            where_conditions = [f"n.{k} = ${k}" for k in conditions.keys()]
            where_clause = f"WHERE {' AND '.join(where_conditions)}"

        query = f"MATCH (n:{label}) {where_clause} RETURN n"
        result = self.db_manager.execute_query(query, conditions or {})

        return [record["n"].as_dict() for record in result]

    def create_relationship(
        self,
        from_label: str,
        from_properties: Dict[str, Any],
        to_label: str,
        to_properties: Dict[str, Any],
        relationship_type: str,
        relationship_properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        2つのノード間にリレーションシップを作成

        Args:
            from_label: 開始ノードのラベル
            from_properties: 開始ノードの識別プロパティ
            to_label: 終了ノードのラベル
            to_properties: 終了ノードの識別プロパティ
            relationship_type: リレーションシップの種類
            relationship_properties: リレーションシップのプロパティ

        Returns:
            作成されたリレーションシップの情報
        """
        # 開始ノードと終了ノードのマッチング条件を構築
        from_match = " AND ".join([f"a.{k} = ${k}_from" for k in from_properties.keys()])
        to_match = " AND ".join([f"b.{k} = ${k}_to" for k in to_properties.keys()])

        # パラメータを構築
        params = {}
        for k, v in from_properties.items():
            params[f"{k}_from"] = v
        for k, v in to_properties.items():
            params[f"{k}_to"] = v

        if relationship_properties:
            params["rel_props"] = relationship_properties

        # リレーションシップ作成クエリ
        rel_props = " $rel_props" if relationship_properties else ""
        query = f"""
        MATCH (a:{from_label}), (b:{to_label})
        WHERE {from_match} AND {to_match}
        CREATE (a)-[r:{relationship_type}{rel_props}]->(b)
        RETURN r
        """

        result = self.db_manager.execute_query(query, params)
        return result.single()[0].as_dict()


# モジュールレベルの関数とグローバル変数
_neo4j_manager: Optional[Neo4jDatabaseManager] = None


def init_neo4j(uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None) -> Neo4jDatabaseManager:
    """
    Neo4jデータベース接続を初期化

    Args:
        uri: Neo4jデータベースのURI
        username: 接続ユーザー名
        password: 接続パスワード

    Returns:
        初期化されたNeo4jDatabaseManager
    """
    global _neo4j_manager
    _neo4j_manager = Neo4jDatabaseManager(uri, username, password)
    return _neo4j_manager


def get_neo4j_driver(self):  # -> 'Driver':
    """
    Neo4jドライバーインスタンスを取得

    Returns:
        Neo4jドライバーインスタンス
    """
    global _neo4j_manager
    if _neo4j_manager is None:
        _neo4j_manager = Neo4jDatabaseManager()
    return _neo4j_manager.driver


@contextmanager
def get_neo4j_session():  # -> PyGenerator['Session', None, None]:
    """
    Neo4jセッションのコンテキストマネージャーを取得

    使用例:
        with get_neo4j_session() as session:
            result = session.run("MATCH (n) RETURN n LIMIT 5")

    Yields:
        Neo4jセッション
    """
    global _neo4j_manager
    if _neo4j_manager is None:
        _neo4j_manager = Neo4jDatabaseManager()

    with _neo4j_manager.session() as session:
        yield session