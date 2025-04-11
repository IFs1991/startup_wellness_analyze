#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neo4jデータベース操作の例を示すサンプルコード

このモジュールはNeo4jデータベースを使用した基本操作のサンプルを提供します。
グラフデータモデルを活用したネットワーク分析や関係性の保存、検索の例を含みます。
"""

import logging
import os
import sys
from typing import Dict, List, Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from backend.database.connection import Neo4jService, get_neo4j_driver, init_db


logger = logging.getLogger(__name__)


def create_startup_network_example():
    """スタートアップ企業間のネットワークモデルを作成する例"""
    try:
        # Neo4j初期化
        init_db()

        # スタートアップノードの作成
        startup_ids = {}
        startups = [
            {"name": "TechVenture", "sector": "AI", "founded_year": 2019},
            {"name": "HealthInnovate", "sector": "HealthTech", "founded_year": 2018},
            {"name": "GreenSolutions", "sector": "CleanTech", "founded_year": 2020},
            {"name": "FinanceHub", "sector": "FinTech", "founded_year": 2017},
            {"name": "EduTech", "sector": "EdTech", "founded_year": 2021}
        ]

        for startup in startups:
            node_id = Neo4jService.create_node("Startup", startup)
            startup_ids[startup["name"]] = node_id
            print(f"スタートアップノード作成: {startup['name']} (ID: {node_id})")

        # 関係の作成
        relationships = [
            ("TechVenture", "HealthInnovate", "COLLABORATES_WITH", {"joint_projects": 2, "strength": 0.8}),
            ("TechVenture", "FinanceHub", "PROVIDES_SERVICE_TO", {"service": "AI分析", "contract_value": 5000000}),
            ("HealthInnovate", "GreenSolutions", "SHARES_OFFICE_WITH", {"location": "渋谷", "since": "2021-04"}),
            ("GreenSolutions", "EduTech", "INVESTS_IN", {"amount": 10000000, "equity": 0.15}),
            ("FinanceHub", "HealthInnovate", "FUNDS", {"amount": 50000000, "equity": 0.2}),
            ("EduTech", "TechVenture", "USES_TECHNOLOGY_FROM", {"technology": "自然言語処理", "license_fee": 2000000})
        ]

        for start, end, rel_type, props in relationships:
            success = Neo4jService.create_relationship(
                startup_ids[start],
                startup_ids[end],
                rel_type,
                props
            )
            if success:
                print(f"関係作成: {start} -> {rel_type} -> {end}")
            else:
                print(f"関係作成失敗: {start} -> {rel_type} -> {end}")

        # 検索例: AIセクターのスタートアップを検索
        ai_startups = Neo4jService.find_nodes("Startup", {"sector": "AI"})
        print(f"\nAIセクターのスタートアップ: {ai_startups}")

        # 高度なクエリ例: 投資関係の検索
        query = """
        MATCH (a:Startup)-[r:INVESTS_IN|FUNDS]->(b:Startup)
        RETURN a.name as investor, b.name as investee, type(r) as relationship,
               r.amount as amount, r.equity as equity
        """
        with get_neo4j_driver() as driver:
            results = driver.execute_query(query)
            print("\n投資関係:")
            for record in results:
                print(f"{record['investor']} -> {record['investee']}: "
                      f"{record['amount']}円 ({record['equity'] * 100}% 株式)")

        # コミュニティ検索例: 関連企業のクラスター
        query = """
        MATCH (s:Startup)
        OPTIONAL MATCH (s)-[r]-(related:Startup)
        WITH s, count(related) AS connections
        RETURN s.name AS startup, s.sector AS sector, connections
        ORDER BY connections DESC
        """
        with get_neo4j_driver() as driver:
            results = driver.execute_query(query)
            print("\n企業間の関連性:")
            for record in results:
                print(f"{record['startup']} ({record['sector']}): {record['connections']}つの関連")

    except Exception as e:
        logger.error(f"エラー発生: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_startup_network_example()
    print("\n完了: Neo4jの例を実行しました。")