# Upload bytes to S3 and return the S3 URI

import boto3, os, uuid
from ..config import settings


session = boto3.Session(
    region_name=settings.AWS_REGION,
    profile_name=settings.AWS_PROFILE  # will be None if not set
)
s3 = session.client("s3")

def _make_key(filename: str) -> str:
    return f"uploads/{uuid.uuid4()}-{os.path.basename(filename)}"

def put_fileobj(fileobj, filename: str, mime: str) -> str:
    """
    Stream a file-like object to S3 without buffering the whole thing in RAM.
    """
    key = _make_key(filename)
    s3.upload_fileobj(
        Fileobj=fileobj,
        Bucket=settings.S3_BUCKET,
        Key=key,
        ExtraArgs={"ContentType": mime},
    )
    return f"s3://{settings.S3_BUCKET}/{key}"


def put_bytes(filename: str, data: bytes, mime: str) -> str:
    """
    Upload raw bytes to S3 and return an s3:// URI.
    """
    key = f"raw/{uuid.uuid4()}-{filename}"
    s3.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=data, ContentType=mime)
    return f"s3://{settings.S3_BUCKET}/{key}"

def get_bytes(s3_uri: str) -> bytes:
    """
    Read the object bytes back (for parsing).
    """
    assert s3_uri.startswith("s3://")
    _, rest = s3_uri.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()