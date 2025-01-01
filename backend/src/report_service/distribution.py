from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

class EmailConfig(BaseModel):
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool = True
    from_email: EmailStr

class EmailTemplate(BaseModel):
    subject: str
    body: str
    variables: Dict[str, Any] = {}

class StorageConfig(BaseModel):
    storage_type: str  # s3, gcs, local
    bucket_name: Optional[str] = None
    base_path: str
    credentials: Dict[str, Any] = {}

class DistributionResult(BaseModel):
    success: bool
    method: str
    recipient: str
    timestamp: datetime
    error_message: Optional[str] = None

class ReportDistributor:
    def __init__(self, config: Dict[str, Any]):
        self.email_config = EmailConfig(**config.get("email", {}))
        self.storage_config = StorageConfig(**config.get("storage", {}))
        self._smtp_client = None

    async def initialize(self):
        """SMTPクライアントを初期化する"""
        if not self._smtp_client:
            self._smtp_client = aiosmtplib.SMTP(
                hostname=self.email_config.smtp_host,
                port=self.email_config.smtp_port,
                use_tls=self.email_config.use_tls
            )
            await self._smtp_client.connect()
            await self._smtp_client.login(
                self.email_config.username,
                self.email_config.password
            )

    async def cleanup(self):
        """SMTPクライアントをクリーンアップする"""
        if self._smtp_client:
            await self._smtp_client.quit()
            self._smtp_client = None

    async def send_report_email(
        self,
        report_content: bytes,
        report_format: str,
        recipients: List[EmailStr],
        template: EmailTemplate,
        attachment_name: Optional[str] = None
    ) -> List[DistributionResult]:
        """レポートをメールで送信する"""
        if not self._smtp_client:
            await self.initialize()

        results = []
        for recipient in recipients:
            try:
                # メールメッセージを作成
                message = MIMEMultipart()
                message["From"] = self.email_config.from_email
                message["To"] = recipient
                message["Subject"] = template.subject

                # 本文を追加
                body = MIMEText(template.body, "html")
                message.attach(body)

                # 添付ファイルを追加
                if attachment_name:
                    attachment = MIMEApplication(report_content)
                    attachment.add_header(
                        "Content-Disposition",
                        "attachment",
                        filename=f"{attachment_name}.{report_format}"
                    )
                    message.attach(attachment)

                # メールを送信
                await self._smtp_client.send_message(message)

                results.append(
                    DistributionResult(
                        success=True,
                        method="email",
                        recipient=recipient,
                        timestamp=datetime.utcnow()
                    )
                )

            except Exception as e:
                results.append(
                    DistributionResult(
                        success=False,
                        method="email",
                        recipient=recipient,
                        timestamp=datetime.utcnow(),
                        error_message=str(e)
                    )
                )

        return results

    async def store_report(
        self,
        report_content: bytes,
        report_format: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DistributionResult:
        """レポートをストレージに保存する"""
        try:
            storage_path = self._generate_storage_path(filename, report_format)

            if self.storage_config.storage_type == "s3":
                await self._store_to_s3(
                    report_content,
                    storage_path,
                    metadata
                )
            elif self.storage_config.storage_type == "gcs":
                await self._store_to_gcs(
                    report_content,
                    storage_path,
                    metadata
                )
            elif self.storage_config.storage_type == "local":
                await self._store_to_local(
                    report_content,
                    storage_path,
                    metadata
                )
            else:
                raise ValueError(f"Unsupported storage type: {self.storage_config.storage_type}")

            return DistributionResult(
                success=True,
                method="storage",
                recipient=storage_path,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            return DistributionResult(
                success=False,
                method="storage",
                recipient=filename,
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )

    async def _store_to_s3(
        self,
        content: bytes,
        storage_path: str,
        metadata: Optional[Dict[str, Any]]
    ):
        """S3にレポートを保存する"""
        import boto3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=self.storage_config.credentials.get('aws_access_key_id'),
            aws_secret_access_key=self.storage_config.credentials.get('aws_secret_access_key'),
            region_name=self.storage_config.credentials.get('region_name')
        )

        s3_client.put_object(
            Bucket=self.storage_config.bucket_name,
            Key=storage_path,
            Body=content,
            Metadata=metadata or {}
        )

    async def _store_to_gcs(
        self,
        content: bytes,
        storage_path: str,
        metadata: Optional[Dict[str, Any]]
    ):
        """Google Cloud Storageにレポートを保存する"""
        # TODO: GCSへの保存処理を実装
        pass

    async def _store_to_local(
        self,
        content: bytes,
        storage_path: str,
        metadata: Optional[Dict[str, Any]]
    ):
        """ローカルストレージにレポートを保存する"""
        import os
        import json

        # 保存先ディレクトリを作成
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # コンテンツを保存
        with open(storage_path, 'wb') as f:
            f.write(content)

        # メタデータを保存
        if metadata:
            metadata_path = f"{storage_path}.metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

    def _generate_storage_path(self, filename: str, format: str) -> str:
        """ストレージパスを生成する"""
        from datetime import datetime
        date_prefix = datetime.utcnow().strftime('%Y/%m/%d')
        return f"{self.storage_config.base_path}/{date_prefix}/{self._generate_id()}_{filename}.{format}"

    def _generate_id(self) -> str:
        """IDを生成する"""
        import uuid
        return str(uuid.uuid4())