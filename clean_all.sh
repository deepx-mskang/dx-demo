#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)

usage() {
    cat <<EOF
Usage: $(basename "$0") [--help]

Remove C++ build artifacts under this repository:
  - build/ next to each demo build.sh
  - bin/ install output (e.g. yolo-multi)
  - cmake-build-* IDE out-of-source directories

Does not remove workspace/, .cache/, assets, or Python virtual environments.
EOF
}

while (( $# )); do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

mapfile -t build_scripts < <(
    find "$REPO_ROOT" -name "build.sh" \
        -not -path "*/clip-multi/*" \
        -not -path "*/launcher/*" \
        -not -path "*/scripts/*" \
        | sort
)

removed=()

remove_path() {
    local path="$1"
    local label="$2"

    if [ -e "$path" ]; then
        rm -rf "$path"
        removed+=("$label")
    fi
}

for script in "${build_scripts[@]}"; do
    project_dir=$(dirname "$script")
    rel="${project_dir#$REPO_ROOT/}"

    remove_path "${project_dir}/build" "${rel}/build"
    remove_path "${project_dir}/bin" "${rel}/bin"
done

while IFS= read -r dir; do
    rel="${dir#$REPO_ROOT/}"
    rm -rf "$dir"
    removed+=("$rel")
done < <(
    find "$REPO_ROOT" -type d -name 'cmake-build-*' -not -path '*/.git/*' 2>/dev/null | sort
)

echo "========================================"
echo "Clean summary"
echo "========================================"

if [ ${#build_scripts[@]} -eq 0 ] && [ ${#removed[@]} -eq 0 ]; then
    echo "No build targets found."
    exit 0
fi

if [ ${#removed[@]} -eq 0 ]; then
    echo "Nothing to clean."
    exit 0
fi

for path in "${removed[@]}"; do
    echo "  removed  $path"
done

echo ""
echo "Removed ${#removed[@]} path(s)."
