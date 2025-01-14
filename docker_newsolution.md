title: Docker接続問題のデバッグガイド
version: 1.0
author: Claude
date: 2024-12-24

problem:
  summary: フロントエンドからバックエンドへのDocker接続確立の問題
  current_state:
    - フロントエンドコンテナからバックエンドコンテナへの接続が不可
    - 環境変数の設定が不適切
    - ポート設定は正しいが通信未確立

analysis:
  root_causes:
    primary:
      - name: 環境変数の不適切な設定
        description: VITE_API_URLがlocalhostを指定しており、コンテナ環境に適していない
      - name: ネットワーク設定の不足
        description: コンテナ間通信のための明示的なネットワーク設定が欠如
    secondary:
      - name: ヘルスチェック機能の欠如
        description: サービスの健全性監視機能が未実装

solutions:
  immediate_fixes:
    environment_variables:
      file: frontend/.env.local
      changes:
        - old: VITE_API_URL=http://localhost:8000
          new: VITE_API_URL=http://backend:8000

    docker_compose:
      file: docker-compose.yml
      changes:
        frontend_service:
          environment:
            - VITE_API_URL=http://backend:8000

  optimizations:
    network_config:
      add_section: |
        networks:
          app-network:
            driver: bridge

      service_updates:
        - service: backend
          network: app-network
        - service: frontend
          network: app-network
        - service: db
          network: app-network

    health_checks:
      backend_service: |
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
          interval: 30s
          timeout: 10s
          retries: 3

verification:
  steps:
    - name: 設定の適用
      commands:
        - docker-compose down
        - docker-compose up --build

    - name: 通信テスト
      commands:
        - docker-compose exec frontend curl http://backend:8000

    - name: ネットワーク確認
      commands:
        - docker network ls
        - docker-compose ps
        - docker network inspect startup_wellness_analyze_app-network

additional_checks:
  if_issues_persist:
    - name: CORS設定の確認
      scope: バックエンドサービス
    - name: ファイアウォール設定
      scope: ホストマシンとコンテナ
    - name: ログ確認
      commands:
        - docker-compose logs backend
        - docker-compose logs frontend

expected_outcome:
  success_criteria:
    - フロントエンドからバックエンドへのAPI通信が確立
    - すべてのサービスが正常に起動
    - ヘルスチェックが正常に動作

  monitoring:
    - コンテナのログ監視
    - ネットワーク接続の定期的な確認
    - APIエンドポイントの応答確認

documentation_updates:
  recommended:
    - プロジェクトのREADME.mdにデバッグ手順を追加
    - 環境構築手順の更新
    - トラブルシューティングガイドの作成

maintenance:
  periodic_checks:
    - ログの定期確認
    - コンテナの健全性監視
    - ネットワーク接続の確認