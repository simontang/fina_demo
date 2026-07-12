from app.config import Settings
from app.storage import S3Storage


def test_presigned_url_uses_public_base_url_when_configured():
    settings = Settings(
        _env_file=None,
        object_storage_endpoint="http://document-minio:9000",
        object_storage_bucket="documents",
        object_storage_access_key="minio",
        object_storage_secret_key="minio123",
        public_presign_base_url="https://files.example.com",
    )

    url = S3Storage(settings).presigned_get_url("assets/asset_1/file.docx")

    assert url.startswith("https://files.example.com/")
    assert "assets/asset_1/file.docx" in url
