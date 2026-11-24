#!/bin/sh
set -eu
exec python "${TRAIN_SCRIPT:-/work/train.py}" "$@"
