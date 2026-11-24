import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from settings import configure_environment


def location(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("Expected s3://bucket/nonempty-prefix")
    return parsed.netloc, parsed.path.strip("/")


def client():
    configure_environment()
    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
                        region_name=os.environ.get("AWS_DEFAULT_REGION", "kr-west1"),
                        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                      request_checksum_calculation="when_required",
                                      response_checksum_validation="when_required",
                                      connect_timeout=20, read_timeout=120,
                                      retries={"max_attempts": 3}))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(uri, root):
    bucket, prefix = location(uri)
    root.mkdir(parents=True, exist_ok=True)
    connection = client()
    connection.download_file(bucket, prefix + "/manifest.json", str(root / "manifest.json"))
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if not name.startswith(("model/", "dataset/")):
            continue
        target = root / name
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("Invalid input manifest path")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == expected["size"] and digest(target) == expected["sha256"]:
            continue
        print("Downloading", name, flush=True)
        connection.download_file(bucket, prefix + "/" + name, str(target))
        if target.stat().st_size != expected["size"] or digest(target) != expected["sha256"]:
            raise ValueError("Input checksum mismatch: " + name)
    return manifest


def upload(uri, root):
    bucket, prefix = location(uri)
    connection = client()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        key = prefix + "/" + path.relative_to(root).as_posix()
        print("Uploading", key, flush=True)
        connection.upload_file(str(path), bucket, key)
        if connection.head_object(Bucket=bucket, Key=key)["ContentLength"] != path.stat().st_size:
            raise ValueError("Output size mismatch: " + key)
