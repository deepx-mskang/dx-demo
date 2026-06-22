#!/bin/bash

language="en"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --language)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --language" >&2
                exit 2
            fi
            language="$2"
            shift 2
            ;;
        --language=*)
            language="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

./kill_automotive.sh

cd ~/dx-demos/automotive/sfa3d

./build/sfa3d_async --full_screen --exit-btn --loop --precompute-bev --language "$language"
