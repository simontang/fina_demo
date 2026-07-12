from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import boto3
from botocore.client import Config

from app.config import Settings


@dataclass(frozen=True)
class StoredObject:
    key: str
    size_bytes: int


class StorageError(RuntimeError):
    pass


class S3Storage:
    def __init__(self, settings: Settings):
        if not settings.object_storage_access_key or not settings.object_storage_secret_key:
            raise StorageError("object storage credentials are required")
        self.settings = settings
        self.bucket = settings.object_storage_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.object_storage_endpoint,
            region_name=settings.object_storage_region,
            aws_access_key_id=settings.object_storage_access_key,
            aws_secret_access_key=settings.object_storage_secret_key,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path" if settings.object_storage_force_path_style else "virtual"},
            ),
        )

    def upload_bytes(self, key: str, data: bytes, content_type: str | None = None) -> StoredObject:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra_args)
        return StoredObject(key=key, size_bytes=len(data))

    def download_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def presigned_get_url(self, key: str, expires_in: int | None = None) -> str:
        expires = expires_in or self.settings.presigned_url_ttl_seconds
        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )
        return self._rewrite_public_base(url)

    def _rewrite_public_base(self, url: str) -> str:
        if not self.settings.public_presign_base_url:
            return url
        parsed_url = urlparse(url)
        parsed_base = urlparse(self.settings.public_presign_base_url)
        return urlunparse(
            (
                parsed_base.scheme or parsed_url.scheme,
                parsed_base.netloc,
                parsed_url.path,
                parsed_url.params,
                parsed_url.query,
                parsed_url.fragment,
            )
        )
