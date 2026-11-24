"""Publish complete Trainer checkpoints and final outputs to S3-compatible storage."""
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from settings import s3_client

REQUIRED_CHECKPOINT_FILES = ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')
MODEL_FILES = ('adapter_model.safetensors', 'adapter_model.bin', 'model.safetensors', 'pytorch_model.bin')
CHECKPOINT_NAME = re.compile(r'checkpoint-[1-9][0-9]*\Z')


def location(uri):
    parsed = urlparse(uri)
    prefix = parsed.path.strip('/')
    if parsed.scheme != 's3' or not parsed.netloc or not prefix or parsed.query or parsed.fragment:
        raise ValueError('Expected s3://bucket/nonempty-prefix')
    if any(part in ('', '.', '..') for part in prefix.split('/')):
        raise ValueError('Invalid S3 prefix')
    return parsed.netloc, prefix


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def file_manifest(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {'size': path.stat().st_size, 'sha256': digest(path)}
    return files


def validate_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError('Invalid checkpoint file path')
    return path


def complete_checkpoint(path):
    return (path.is_dir() and CHECKPOINT_NAME.fullmatch(path.name) is not None
            and not (path / 'checkpoint-is-incomplete.txt').exists()
            and all((path / name).is_file() for name in REQUIRED_CHECKPOINT_FILES)
            and any((path / name).is_file() for name in MODEL_FILES))


def latest_local_checkpoint(root):
    found = [path for path in root.glob('checkpoint-*') if complete_checkpoint(path)]
    return max(found, key=lambda path: int(path.name.rsplit('-', 1)[1])) if found else None


class S3CheckpointStore:
    def __init__(self, uri, client=None):
        self.bucket, self.prefix = location(uri)
        self.client = client or s3_client()

    def key(self, name):
        return self.prefix + '/' + name

    def publish(self, checkpoint, fingerprint):
        checkpoint = Path(checkpoint)
        if not complete_checkpoint(checkpoint):
            raise ValueError('Checkpoint is incomplete: ' + str(checkpoint))
        files = file_manifest(checkpoint)
        manifest = {'version': 1, 'checkpoint': checkpoint.name,
                    'step': int(checkpoint.name.rsplit('-', 1)[1]),
                    'fingerprint': fingerprint, 'files': files}
        for name, info in files.items():
            key = self.key(checkpoint.name + '/' + name)
            self.client.upload_file(str(checkpoint / name), self.bucket, key)
            if self.client.head_object(Bucket=self.bucket, Key=key)['ContentLength'] != info['size']:
                raise ValueError('Uploaded checkpoint size mismatch: ' + name)
        self.client.put_object(Bucket=self.bucket, Key=self.key(checkpoint.name + '/manifest.json'),
                               Body=json.dumps(manifest, sort_keys=True).encode())
        # This pointer is written only after all files and their manifest exist.
        pointer = {key: manifest[key] for key in ('version', 'checkpoint', 'step', 'fingerprint')}
        self.client.put_object(Bucket=self.bucket, Key=self.key('latest.json'),
                               Body=json.dumps(pointer, sort_keys=True).encode())
        return manifest

    def restore_latest(self, root, fingerprint):
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key('latest.json'))
        except ClientError as error:
            if error.response.get('Error', {}).get('Code') in ('NoSuchKey', '404'):
                return None
            raise
        pointer = json.loads(response['Body'].read())
        name = pointer.get('checkpoint')
        if pointer.get('version') != 1 or not isinstance(name, str) or not CHECKPOINT_NAME.fullmatch(name):
            raise ValueError('Invalid latest checkpoint pointer')
        if pointer.get('step') != int(name.rsplit('-', 1)[1]):
            raise ValueError('Latest pointer step does not match checkpoint name')
        if pointer.get('fingerprint') != fingerprint:
            raise ValueError('Checkpoint fingerprint differs from this training configuration')
        manifest_response = self.client.get_object(Bucket=self.bucket, Key=self.key(name + '/manifest.json'))
        manifest = json.loads(manifest_response['Body'].read())
        if (manifest.get('version') != 1 or manifest.get('checkpoint') != name
                or manifest.get('step') != pointer.get('step')
                or manifest.get('fingerprint') != fingerprint
                or not isinstance(manifest.get('files'), dict)):
            raise ValueError('Checkpoint manifest does not match latest pointer')
        files = manifest['files']
        if (not all(item in files for item in REQUIRED_CHECKPOINT_FILES)
                or not any(item in files for item in MODEL_FILES)):
            raise ValueError('Checkpoint manifest is missing Trainer or model state')
        for filename, info in files.items():
            validate_relative(filename)
            if (not isinstance(info, dict) or not isinstance(info.get('size'), int)
                    or info['size'] < 0 or not isinstance(info.get('sha256'), str)
                    or not re.fullmatch(r'[0-9a-f]{64}', info['sha256'])):
                raise ValueError('Invalid checkpoint file metadata')
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        target = root / name
        if target.exists():
            if complete_checkpoint(target) and file_manifest(target) == files:
                return target
            raise ValueError('Local checkpoint conflicts with S3: ' + str(target))
        temporary = root / ('.' + name + '.partial-' + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            for filename, info in files.items():
                relative = validate_relative(filename)
                path = temporary.joinpath(*relative.parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, self.key(name + '/' + filename), str(path))
                if path.stat().st_size != info['size'] or digest(path) != info['sha256']:
                    raise ValueError('Checkpoint file verification failed: ' + filename)
            if (not all((temporary / item).is_file() for item in REQUIRED_CHECKPOINT_FILES)
                    or not any((temporary / item).is_file() for item in MODEL_FILES)):
                raise ValueError('Downloaded checkpoint is incomplete')
            temporary.rename(target)
            return target
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def upload_results(uri, root, client=None):
    bucket, prefix = location(uri)
    connection = client or s3_client()
    root = Path(root)
    files = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or 'checkpoints' in path.relative_to(root).parts:
            continue
        name = path.relative_to(root).as_posix()
        files[name] = {'size': path.stat().st_size, 'sha256': digest(path)}
        key = prefix + '/' + name
        connection.upload_file(str(path), bucket, key)
        if connection.head_object(Bucket=bucket, Key=key)['ContentLength'] != files[name]['size']:
            raise ValueError('Uploaded result size mismatch: ' + name)
    connection.put_object(Bucket=bucket, Key=prefix + '/result-manifest.json',
                          Body=json.dumps({'version': 1, 'files': files}, sort_keys=True).encode())
    return files


def download_results(uri, destination, client=None):
    """Restore a completed result set only after its commit manifest is present."""
    bucket, prefix = location(uri)
    connection = client or s3_client()
    response = connection.get_object(Bucket=bucket, Key=prefix + '/result-manifest.json')
    manifest = json.loads(response['Body'].read())
    files = manifest.get('files')
    if manifest.get('version') != 1 or not isinstance(files, dict):
        raise ValueError('Invalid result manifest')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Result destination already exists: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name('.' + destination.name + '.partial-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        for name, info in files.items():
            relative = validate_relative(name)
            if (not isinstance(info, dict) or not isinstance(info.get('size'), int)
                    or info['size'] < 0 or not isinstance(info.get('sha256'), str)
                    or not re.fullmatch(r'[0-9a-f]{64}', info['sha256'])):
                raise ValueError('Invalid result file metadata')
            path = temporary.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            connection.download_file(bucket, prefix + '/' + name, str(path))
            if path.stat().st_size != info['size'] or digest(path) != info['sha256']:
                raise ValueError('Result file verification failed: ' + name)
        temporary.rename(destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
