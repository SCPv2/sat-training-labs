"""Runtime configuration for AWS S3 or an explicit S3-compatible endpoint."""
import os
from urllib.parse import urlparse

import boto3
from botocore.config import Config


def s3_client():
    endpoint = os.environ.get('S3_ENDPOINT_URL') or None
    region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or 'us-east-1'
    options = {'endpoint_url': endpoint, 'region_name': region,
               'config': Config(signature_version='s3v4',
                                connect_timeout=20, read_timeout=120,
                                retries={'max_attempts': 3})}
    if endpoint:
        host = urlparse(endpoint).hostname
        if not host or urlparse(endpoint).scheme not in ('http', 'https'):
            raise ValueError('S3_ENDPOINT_URL must be an HTTP(S) endpoint')
        options['config'] = Config(signature_version='s3v4',
                                   s3={'addressing_style': 'path'},
                                   connect_timeout=20, read_timeout=120,
                                   retries={'max_attempts': 3})
        for name in ('NO_PROXY', 'no_proxy'):
            entries = [item.strip() for item in os.environ.get(name, '').split(',') if item.strip()]
            if host not in entries:
                entries.append(host)
            os.environ[name] = ','.join(entries)
    # Boto3 resolves EC2 instance roles, task roles, or environment credentials.
    # No account key is embedded in the training image or source code.
    return boto3.client('s3', **options)


def storage_uri(cli_value, environment_name):
    return cli_value or os.environ.get(environment_name) or None
