スタートアップウェルネス分析プラットフォームの設定修正まとめ
修正した問題点
PostgreSQLのバージョン互換性
データディレクトリがPostgreSQL 15で作成されていたが、使用イメージはPostgreSQL 14だった
PostgreSQLイメージをバージョン15に更新して解決
バックエンドのSQLAlchemy依存関係
バックエンドが起動時にSQLAlchemyを見つけられずエラーになっていた
environment.ymlに必要なライブラリを追加:
SQLAlchemy
psycopg2
alembic
entrypoint.shスクリプトの強化
起動時に必要な依存関係を自動的にチェックするコードを追加
SQLAlchemyとpsycopg2が存在しなければインストールするロジックを実装
フロントエンドのコンテナ設定
npmコマンドが見つからないエラーが発生
エントリーポイントを明示的に設定してnpmコマンドを避けるように修正
docker-compose.ymlの修正
サービス間の依存関係を適切に設定
ヘルスチェック設定の調整
container_nameを削除し自動生成名を使用するように変更
最終的な構成
バックエンド環境設定
Apply to docker-compo...
entrypoint.sh (バックエンド)
Apply to docker-compo...
Run
フロントエンドDockerfile
Apply to docker-compo...
docker-compose.yml
Apply to docker-compo...
シンプルなフロントエンド (代替案)
Apply to docker-compo...
動作確認結果
PostgreSQLとRedisは安定して動作している
バックエンドの依存関係問題は解決したが、まだ一部問題が残っている
フロントエンドは複雑なnodejs/ビルド問題があるため、単純なNginxコンテナで代替可能
今後の課題
バックエンドの残りの問題を調査 (起動時のエラーログを確認)
フロントエンドのマルチステージビルドの問題解決
docker-compose.override.ymlとdocker-compose.ymlの適切な連携