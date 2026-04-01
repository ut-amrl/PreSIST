#!/bin/bash
set -euo pipefail

EXPLICIT_TAG=""
FORCE_FROM_SCRATCH=0
SHOULD_TIMESTAMP_TAG=0
KEEP_ARCHIVE=0
TRY_LOCAL_FIRST=1

BUILD_TOOL="${BUILD_TOOL:-podman}"
IMAGE_NAME="${IMAGE_NAME:-image}"
PRIMARY_REMOTE_HOST="${PRIMARY_REMOTE_HOST:-robolidar}"
FALLBACK_REMOTE_HOST="${FALLBACK_REMOTE_HOST:-}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-$(pwd)}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-Dockerfile}"

LOCAL_ARCHIVE_DIR="${LOCAL_ARCHIVE_DIR:-$HOME/workspaces/dockerimages}"
REMOTE_ARCHIVE_DIR="${REMOTE_ARCHIVE_DIR:-$HOME/workspaces/dockerimages}"

usage() {
    local prog_name
    prog_name="${PROGRAM_NAME:-$(basename "$0")}"
    cat <<EOF
Usage: $prog_name [options]

Generic local Docker/Podman build with remote fallback.

Options:
  --image-name <name>             Base image name (default: \$IMAGE_NAME)
  --dockerfile <path>             Dockerfile path relative to repo root/cwd
  --build-tool <tool>             Build CLI (default: \$BUILD_TOOL, e.g. podman|docker)
  -f, --force                     Build with --no-cache
  --timestamp-tag                 Tag as <image-name>:<timestamp> instead of latest
  --tag <tag>                     Explicit image tag
  --keep-archive                  Keep local and remote OCI archives
  --no-local                      Skip local build attempt, go straight to remote
  --remote-host <host>            Primary remote build host (default: \$PRIMARY_REMOTE_HOST)
  --fallback-remote-host <host>   Optional fallback remote host if primary fails
  --remote-project-dir <path>     Remote repo/project directory
  -h, --help                      Show this help
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --image-name) IMAGE_NAME="$2"; shift ;;
        --dockerfile) DOCKERFILE_PATH="$2"; shift ;;
        --build-tool) BUILD_TOOL="$2"; shift ;;
        -f|--force) FORCE_FROM_SCRATCH=1 ;;
        --timestamp-tag) SHOULD_TIMESTAMP_TAG=1 ;;
        --keep-archive) KEEP_ARCHIVE=1 ;;
        --no-local) TRY_LOCAL_FIRST=0 ;;
        --remote-host) PRIMARY_REMOTE_HOST="$2"; shift ;;
        --fallback-remote-host) FALLBACK_REMOTE_HOST="$2"; shift ;;
        --remote-project-dir) REMOTE_PROJECT_DIR="$2"; shift ;;
        --tag) EXPLICIT_TAG="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown parameter: $1"; usage; exit 1 ;;
    esac
    shift
done

mkdir -p "$LOCAL_ARCHIVE_DIR"

if [ -n "$EXPLICIT_TAG" ]; then
    TAG="$EXPLICIT_TAG"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    if [ "$SHOULD_TIMESTAMP_TAG" -eq 1 ]; then
        TAG="${IMAGE_NAME}:${TIMESTAMP}"
    else
        TAG="${IMAGE_NAME}:latest"
    fi
fi

SAFE_TAG="${TAG//[:\/]/_}"
REMOTE_ARCHIVE="${REMOTE_ARCHIVE_DIR}/${SAFE_TAG}.oci.tar"
LOCAL_ARCHIVE="${LOCAL_ARCHIVE_DIR}/${SAFE_TAG}.oci.tar"

build_image_cmd() {
    local dockerfile="$1"
    local tag="$2"
    if [ "$FORCE_FROM_SCRATCH" -eq 1 ]; then
        "$BUILD_TOOL" build --no-cache --network=host -f "$dockerfile" -t "$tag" .
    else
        "$BUILD_TOOL" build --network=host -f "$dockerfile" -t "$tag" .
    fi
}

try_local_build() {
    echo "Attempting local build: $TAG"
    if build_image_cmd "$DOCKERFILE_PATH" "$TAG"; then
        echo "Local build succeeded."
        return 0
    fi
    echo "Local build failed."
    return 1
}

try_remote_build_and_load() {
    local remote_host="$1"
    if [ -z "$remote_host" ]; then
        return 1
    fi

    echo "Attempting remote build on host: $remote_host"
    ssh "$remote_host" "mkdir -p '$REMOTE_ARCHIVE_DIR'"

    local no_cache_flag=""
    if [ "$FORCE_FROM_SCRATCH" -eq 1 ]; then
        no_cache_flag="--no-cache"
    fi

    local remote_build_cmd
    remote_build_cmd="cd '$REMOTE_PROJECT_DIR' && '$BUILD_TOOL' build $no_cache_flag --network=host -f '$DOCKERFILE_PATH' -t '$TAG' ."

    echo "Running remote command on $remote_host: $remote_build_cmd"
    if ! ssh "$remote_host" "$remote_build_cmd"; then
        echo "Remote build failed on $remote_host"
        return 1
    fi

    echo "Saving remote OCI archive..."
    ssh "$remote_host" "'$BUILD_TOOL' save --format oci-archive -o '$REMOTE_ARCHIVE' '$TAG'"

    echo "Copying archive from $remote_host to local machine..."
    scp "$remote_host:$REMOTE_ARCHIVE" "$LOCAL_ARCHIVE"

    echo "Loading image locally..."
    "$BUILD_TOOL" load -i "$LOCAL_ARCHIVE"

    if [ "$SHOULD_TIMESTAMP_TAG" -eq 1 ]; then
        echo "Tagging $TAG as ${IMAGE_NAME}:latest locally"
        "$BUILD_TOOL" tag "$TAG" "${IMAGE_NAME}:latest"
    fi

    if [ "$KEEP_ARCHIVE" -eq 0 ]; then
        echo "Cleaning up build archives..."
        rm -f "$LOCAL_ARCHIVE"
        ssh "$remote_host" "rm -f '$REMOTE_ARCHIVE'"
    else
        echo "Keeping archives:"
        echo "  Local:  $LOCAL_ARCHIVE"
        echo "  Remote: $REMOTE_ARCHIVE (on $remote_host)"
    fi

    echo "Remote build and local load succeeded from $remote_host"
    return 0
}

if [ "$TRY_LOCAL_FIRST" -eq 1 ]; then
    if try_local_build; then
        echo "Done. Image available locally as $TAG"
        exit 0
    fi
fi

if try_remote_build_and_load "$PRIMARY_REMOTE_HOST"; then
    echo "Done. Image available locally as $TAG"
    exit 0
fi

if [ -n "$FALLBACK_REMOTE_HOST" ] && [ "$FALLBACK_REMOTE_HOST" != "$PRIMARY_REMOTE_HOST" ]; then
    echo "Primary remote failed. Trying fallback host: $FALLBACK_REMOTE_HOST"
    if try_remote_build_and_load "$FALLBACK_REMOTE_HOST"; then
        echo "Done. Image available locally as $TAG"
        exit 0
    fi
fi

echo "ERROR: Build failed locally and on all configured remote hosts."
exit 1
