#!/bin/bash

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
clean_args=()

while (( $# )); do
    case "$1" in
        --clean) clean_args=(--clean); shift;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--clean]"; exit 1;;
    esac
done

mapfile -t build_scripts < <(
    find "$REPO_ROOT" -name "build.sh" \
        -not -path "*/clip-multi/*" \
        -not -path "*/launcher/*" \
        -not -path "*/scripts/*" \
        | sort
)

if [ ${#build_scripts[@]} -eq 0 ]; then
    echo "No build.sh scripts found."
    exit 1
fi

passed=()
failed=()

echo "Found ${#build_scripts[@]} build target(s)."
echo ""

for script in "${build_scripts[@]}"; do
    rel="${script#$REPO_ROOT/}"
    echo "========================================"
    echo "Building: $rel"
    echo "========================================"

    if (cd "$(dirname "$script")" && ./build.sh "${clean_args[@]}"); then
        passed+=("$rel")
    else
        failed+=("$rel")
        echo "FAILED: $rel" >&2
    fi
    echo ""
done

echo "========================================"
echo "Build summary (${#passed[@]}/${#build_scripts[@]} succeeded)"
echo "========================================"

for target in "${passed[@]}"; do
    echo "  OK   $target"
done

if [ ${#failed[@]} -gt 0 ]; then
    for target in "${failed[@]}"; do
        echo "  FAIL $target"
    done
    exit 1
fi
