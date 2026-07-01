#!/bin/bash
set -e

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
clean_args=()
lang_cpp=true
lang_python=true

while (( $# )); do
    case "$1" in
        --clean) clean_args=(--clean); shift;;
        --lang)
            if [ "$2" == "cpp" ]; then
                lang_cpp=true
                lang_python=false
            elif [ "$2" == "python" ]; then
                lang_cpp=false
                lang_python=true
            elif [ "$2" == "all" ]; then
                lang_cpp=true
                lang_python=true
            else
                echo "Unknown lang: $2. Use cpp, python, or all."
                exit 1
            fi
            shift 2
            ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--clean] [--lang cpp|python|all]"; exit 1;;
    esac
done

if [ "$lang_python" = true ]; then
    echo "========================================"
    echo "Setting up Python Environment"
    echo "========================================"
    bash "$REPO_ROOT/setup_env.sh"
    echo ""
fi

if [ "$lang_cpp" = true ]; then
    echo "========================================"
    echo "Building C++ Projects"
    echo "========================================"
    mapfile -t build_scripts < <(
        find "$REPO_ROOT/apps" -mindepth 3 -maxdepth 4 -name "build.sh" | grep "/cpp/" | sort
    )

    if [ ${#build_scripts[@]} -eq 0 ]; then
        echo "No build.sh scripts found for C++ projects in apps/ directory."
    else
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
    fi
fi
