#!/bin/bash
set -e

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
clean_args=()
lang_cpp=true
lang_python=true
setup_ocr_web=true
ocr_web_failed=false

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
        --no-ocr-web) setup_ocr_web=false; shift;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--clean] [--lang cpp|python|all] [--no-ocr-web]"
            echo "  --no-ocr-web   Skip the OCR Web environment setup (heavy: clones repos,"
            echo "                 creates two venvs, installs paddlepaddle)"
            exit 1
            ;;
    esac
done

if [ "$lang_python" = true ]; then
    echo "========================================"
    echo "Setting up Python Environment"
    echo "========================================"
    bash "$REPO_ROOT/setup_env.sh"
    echo ""

    if [ "$setup_ocr_web" = true ]; then
        echo "========================================"
        echo "Setting up OCR Web Environment"
        echo "========================================"
        # Clones two repos and builds its own venvs, so it is network-bound and slow.
        # A failure here must not abort the C++ build below; it is reported at the end.
        if bash "$REPO_ROOT/apps/paddle-ocr-web/build.sh" "${clean_args[@]}"; then
            echo "  OK   apps/paddle-ocr-web"
        else
            ocr_web_failed=true
            echo "FAILED: apps/paddle-ocr-web" >&2
            echo "        Retry with apps/paddle-ocr-web/python/build.sh" >&2
        fi
        echo ""
    else
        echo "Skipping the OCR Web environment setup (--no-ocr-web)."
        echo ""
    fi
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
            if [ "$ocr_web_failed" = true ]; then
                echo "  FAIL apps/paddle-ocr-web (OCR Web environment)" >&2
            fi
            exit 1
        fi
    fi
fi

if [ "$ocr_web_failed" = true ]; then
    echo "========================================" >&2
    echo "The OCR Web environment setup failed - the OCR Web demo will not start." >&2
    echo "Retry with apps/paddle-ocr-web/python/build.sh" >&2
    echo "========================================" >&2
    exit 1
fi
