# -*- coding: utf-8 -*-
"""
リポジトリパッケージ
データストアへのアクセスを抽象化するリポジトリクラスを提供します。
"""

from .firestore import FirestoreRepository
from .sql import SQLRepository
from .neo4j import Neo4jRepository
from .factory import ConcreteRepositoryFactory, repository_factory

__all__ = [
    'FirestoreRepository',
    'SQLRepository',
    'Neo4jRepository',
    'ConcreteRepositoryFactory',
    'repository_factory'
]