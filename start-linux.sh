#!/usr/bin/env sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ ! -x .venv/bin/python ]; then
  echo "Please follow README.md to create the Python 3.12 environment first." >&2
  exit 1
fi
exec .venv/bin/python -m assembly_workbench "$@"
