#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
tools_dir="$PWD/.walkthrough-tools"
export UV_PYTHON_INSTALL_DIR="$tools_dir/python"
export UV_CACHE_DIR="$tools_dir/uv-cache"

for dependency in uv npm; do
    command -v "$dependency" >/dev/null || { echo "Install $dependency first." >&2; exit 1; }
done

if [[ ! -x .walkthrough-venv/bin/python ]]; then
    uv venv --python 3.12 .walkthrough-venv
fi
uv -q pip install --python .walkthrough-venv/bin/python -r requirements-walkthrough.txt

brush_dir="$tools_dir/brush"
brush_package=brush-app-aarch64-apple-darwin
if [[ -z ${BRUSH_BIN:-} && ! -x "$brush_dir/$brush_package/brush_app" ]]; then
    if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
        echo 'Set BRUSH_BIN to the Brush 0.3.0 executable for this platform and rerun setup.' >&2
        exit 1
    fi
    mkdir -p "$brush_dir"
    curl -fsSL "https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/$brush_package.tar.xz" \
        -o "$brush_dir/brush.tar.xz"
    printf '%s  %s\n' '65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c' \
        "$brush_dir/brush.tar.xz" | shasum -a 256 -c -
    tar -xJf "$brush_dir/brush.tar.xz" -C "$brush_dir"
    rm "$brush_dir/brush.tar.xz"
fi

viewer_dir="$tools_dir/supersplat"
if [[ ! -f "$viewer_dir/package/public/index.js" ]]; then
    mkdir -p "$viewer_dir"
    archive=$(npm --cache "$tools_dir/npm-cache" pack @playcanvas/supersplat-viewer@1.43.3 \
        --silent --pack-destination "$viewer_dir")
    tar -xzf "$viewer_dir/$archive" -C "$viewer_dir"
    rm "$viewer_dir/$archive"
fi

echo 'Run .walkthrough-venv/bin/python -m pipeline.walkthrough and open http://127.0.0.1:8181/'
