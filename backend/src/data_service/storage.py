from typing import List, Dict, Optional, Any, BinaryIO
from datetime import datetime
from pydantic import BaseModel
import json
import boto3
from botocore.exceptions import ClientError

class StorageMetadata(BaseModel):
    id: str
    filename: str
    content_type: str
    size: int
    checksum: str
    storage_path: str
    created_at: datetime
    updated_at: datetime
    tags: Dict[str, str] = {}

class StorageResult(BaseModel):
    success: bool
    metadata: Optional[StorageMetadata] = None
    error_message: Optional[str] = None

class DataStorage:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.get('aws_access_key_id'),
            aws_secret_access_key=config.get('aws_secret_access_key'),
            region_name=config.get('aws_region')
        )
        self.bucket_name = config.get('bucket_name')

    async def store_data(
        self,
        data: Any,
        filename: str,
        content_type: str,
        tags: Optional[Dict[str, str]] = None
    ) -> StorageResult:
        """データを保存する"""
        try:
            # データをバイナリに変換
            if isinstance(data, (dict, list)):
                binary_data = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                binary_data = data.encode('utf-8')
            elif isinstance(data, bytes):
                binary_data = data
            else:
                raise ValueError("Unsupported data type")

            # チェックサムを計算
            import hashlib
            checksum = hashlib.md5(binary_data).hexdigest()

            # ストレージパスを生成
            storage_path = self._generate_storage_path(filename)

            # S3にアップロード
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=storage_path,
                Body=binary_data,
                ContentType=content_type,
                Metadata={
                    'checksum': checksum,
                    **tags if tags else {}
                }
            )

            # メタデータを作成
            metadata = StorageMetadata(
                id=self._generate_id(),
                filename=filename,
                content_type=content_type,
                size=len(binary_data),
                checksum=checksum,
                storage_path=storage_path,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                tags=tags or {}
            )

            return StorageResult(success=True, metadata=metadata)

        except Exception as e:
            return StorageResult(success=False, error_message=str(e))

    async def retrieve_data(
        self,
        storage_path: str,
        as_binary: bool = False
    ) -> Optional[Any]:
        """データを取得する"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=storage_path
            )

            data = response['Body'].read()

            if as_binary:
                return data

            content_type = response.get('ContentType', '')
            if 'json' in content_type:
                return json.loads(data.decode('utf-8'))
            elif 'text' in content_type:
                return data.decode('utf-8')
            else:
                return data

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    async def backup_data(
        self,
        storage_path: str,
        backup_suffix: Optional[str] = None
    ) -> StorageResult:
        """データをバックアップする"""
        try:
            # 元のデータを取得
            data = await self.retrieve_data(storage_path, as_binary=True)
            if not data:
                return StorageResult(
                    success=False,
                    error_message="Original data not found"
                )

            # バックアップパスを生成
            backup_path = self._generate_backup_path(storage_path, backup_suffix)

            # バックアップを作成
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=backup_path,
                Body=data,
                Metadata={'original_path': storage_path}
            )

            # メタデータを取得
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=backup_path
            )

            metadata = StorageMetadata(
                id=self._generate_id(),
                filename=storage_path.split('/')[-1],
                content_type=response.get('ContentType', ''),
                size=response.get('ContentLength', 0),
                checksum=response.get('ETag', '').strip('"'),
                storage_path=backup_path,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                tags={'backup_of': storage_path}
            )

            return StorageResult(success=True, metadata=metadata)

        except Exception as e:
            return StorageResult(success=False, error_message=str(e))

    async def delete_data(self, storage_path: str) -> bool:
        """データを削除する"""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=storage_path
            )
            return True
        except:
            return False

    async def list_data(
        self,
        prefix: Optional[str] = None,
        max_items: int = 1000
    ) -> List[StorageMetadata]:
        """データ一覧を取得する"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix if prefix else '',
                MaxItems=max_items
            )

            metadata_list = []
            for page in page_iterator:
                for obj in page.get('Contents', []):
                    # オブジェクトのメタデータを取得
                    response = self.s3_client.head_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )

                    metadata = StorageMetadata(
                        id=self._generate_id(),
                        filename=obj['Key'].split('/')[-1],
                        content_type=response.get('ContentType', ''),
                        size=obj['Size'],
                        checksum=obj['ETag'].strip('"'),
                        storage_path=obj['Key'],
                        created_at=obj['LastModified'],
                        updated_at=obj['LastModified'],
                        tags=response.get('Metadata', {})
                    )
                    metadata_list.append(metadata)

            return metadata_list

        except Exception:
            return []

    def _generate_storage_path(self, filename: str) -> str:
        """ストレージパスを生成する"""
        from datetime import datetime
        date_prefix = datetime.utcnow().strftime('%Y/%m/%d')
        return f"{date_prefix}/{self._generate_id()}_{filename}"

    def _generate_backup_path(self, original_path: str, suffix: Optional[str] = None) -> str:
        """バックアップパスを生成する"""
        if suffix:
            return f"{original_path}.{suffix}"
        return f"{original_path}.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    def _generate_id(self) -> str:
        """IDを生成する"""
        import uuid
        return str(uuid.uuid4())