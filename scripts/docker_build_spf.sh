#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SHARED_SCRIPT="$REPO_ROOT/scripts/docker_build_with_fallback.sh"

if [ ! -x "$SHARED_SCRIPT" ]; then
    echo "ERROR: Shared build script not found or not executable: $SHARED_SCRIPT"
    exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-spf}" \
DOCKERFILE_PATH="${DOCKERFILE_PATH:-Dockerfile.spf}" \
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-$HOME/semantic-persistence-filter}" \
PRIMARY_REMOTE_HOST="${PRIMARY_REMOTE_HOST:-tarunrav@robolidar.csres.utexas.edu}" \
PROGRAM_NAME="${PROGRAM_NAME:-$(basename "$0")}" \
"$SHARED_SCRIPT" "$@"
