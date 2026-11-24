"""Defaults for the dev3 SAT demo; runtime environment values take precedence."""

import os
from urllib.parse import urlparse


EMBEDDED_CREDENTIALS = {}

DEFAULTS = {
    "S3_ENDPOINT_URL": "https://object-store.private.kr-west1.dev3.samsungsdscloud.com",
    "AWS_DEFAULT_REGION": "kr-west1",
    "INPUT_URI": "s3://sat-gemma4-demo-20261103-9288a66d/gemma4-support/inputs/v1",
    "OUTPUT_URI": "s3://sat-gemma4-demo-20261103-9288a66d/gemma4-support/runs",
    "WORK_DIR": "/work",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
DEFAULTS.update(EMBEDDED_CREDENTIALS)


def configure_environment():
    for name, value in DEFAULTS.items():
        os.environ.setdefault(name, value)

    # Preserve inherited exclusions and bypass proxies for the selected OBS host.
    exclusions = []
    for name in ("NO_PROXY", "no_proxy"):
        exclusions.extend(item.strip() for item in os.environ.get(name, "").split(",") if item.strip())
    exclusions.extend(["localhost", "127.0.0.1"])
    host = urlparse(os.environ["S3_ENDPOINT_URL"]).hostname
    if host:
        exclusions.append(host)
    value = ",".join(dict.fromkeys(exclusions))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value
