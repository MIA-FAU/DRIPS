#!/usr/bin/env bash
set -euo pipefail
exec python3 /opt/drips/drips_infer.py "$@"
