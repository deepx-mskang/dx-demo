#!/bin/bash
set -e

# Sets up the PP-OCRv5 web demo: Gradio UI (:7860) + PaddleOCR-deepx OCR server (:8080).
# Safe to re-run; every step is skipped when it is already in place.
#
# Usage: ./build.sh [--dx_rt PATH] [--cpu-only] [--clean] [--help]

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

WEB_REPO="https://github.com/DEEPX-AI/PP-OCRv5_Online_demo-deepx.git"
SERVER_REPO="https://github.com/DEEPX-AI/PaddleOCR-deepx.git"
REPO_BRANCH="deepx"

WEB_DIR="${SCRIPT_DIR}/PP-OCRv5_Online_demo-deepx"
SERVER_DIR="${SCRIPT_DIR}/PaddleOCR-deepx"
FASTAPI_DIR="${SERVER_DIR}/deploy/fastapi"
VENV_DIR="${SCRIPT_DIR}/.venv"
BUILD_DIR="${SCRIPT_DIR}/.build"

DX_RT_PATH="${DX_RT_PATH:-}"
cpu_only=false
clean_build=false

usage() {
    echo "Usage: $0 [--dx_rt PATH] [--cpu-only] [--clean]"
    echo "  --dx_rt PATH   dx_rt checkout used to build the dx_engine python binding"
    echo "                 (auto-detected when omitted; DX_RT_PATH env also works)"
    echo "  --cpu-only     Skip the NPU setup (no dx_engine, no .dxnn models)"
    echo "  --clean        Remove the venvs and the dx_engine build dir first"
}

while (( $# )); do
    case "$1" in
        --dx_rt) DX_RT_PATH="$2"; shift 2;;
        --cpu-only) cpu_only=true; shift;;
        --clean) clean_build=true; shift;;
        --help|-h) usage; exit 0;;
        *) echo "Unknown argument: $1"; usage; exit 1;;
    esac
done

if [ "$clean_build" = true ]; then
    echo "Removing existing environments ..."
    rm -rf "${VENV_DIR}" "${FASTAPI_DIR}/venv" "${BUILD_DIR}"
fi

# --- 1. Sources ---------------------------------------------------------------
clone_if_missing() {
    local dir="$1" repo="$2"
    if [ -d "${dir}/.git" ]; then
        echo "Already cloned: $(basename "${dir}")"
    else
        echo "Cloning ${repo} ..."
        git clone --depth 1 --branch "${REPO_BRANCH}" "${repo}" "${dir}"
    fi
}

clone_if_missing "${WEB_DIR}" "${WEB_REPO}"
clone_if_missing "${SERVER_DIR}" "${SERVER_REPO}"

# --- 2. Web UI venv -----------------------------------------------------------
# Kept apart from the shared repo .venv because gradio 5.30.0 is pinned.
#
# requirements.txt pins pillow==9.5.0, which has no wheel for python 3.12+. Which
# interpreter we land on depends on the shell build.sh was started from (running
# it inside the dx-runtime venv gives 3.12), so on some machines pip falls back to
# the pillow sdist: that build needs zlib/libjpeg headers and, without libwebp-dev,
# quietly yields a Pillow with no WebP encoder - which then breaks every Gradio
# gallery render. --only-binary=Pillow turns that into a clean resolution failure
# we can handle, instead of a slow source build with a silently broken result.
install_web_requirements() {
    local req="${WEB_DIR}/requirements.txt"
    if "${VENV_DIR}"/bin/pip install --only-binary=Pillow -r "${req}"; then
        return
    fi
    echo "No Pillow wheel for $("${VENV_DIR}"/bin/python -V); installing a newer Pillow instead."
    local filtered
    filtered="$(mktemp)"
    grep -viE '^[[:space:]]*pillow([[:space:]]*[<>=!~]|$)' "${req}" > "${filtered}"
    "${VENV_DIR}"/bin/pip install -r "${filtered}"
    rm -f "${filtered}"
    # Upper bound comes from gradio 5.30 (pillow<12.0)
    "${VENV_DIR}"/bin/pip install --only-binary=:all: 'pillow>=10.4,<12.0'
}

if [ -x "${VENV_DIR}/bin/python" ]; then
    echo "Already present: web UI venv"
else
    echo "Creating the web UI venv ..."
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}"/bin/pip install --upgrade pip
    install_web_requirements
fi

# --- 2b. Pillow WebP support --------------------------------------------------
# Safety net for the same problem, covering venvs that step 2 did not create:
# environments built before the guard above, and the rare wheel without WebP.
# gradio's Gallery writes WebP by default, so a Pillow lacking the encoder makes
# every result render die with `KeyError: 'WEBP'` after a successful inference.
# Runs on every build.sh, including when the venv above was already present.
pillow_has_webp() {
    "${VENV_DIR}"/bin/python -c \
        'from PIL import features; raise SystemExit(0 if features.check("webp") else 1)' \
        > /dev/null 2>&1
}

if pillow_has_webp; then
    echo "Already present: Pillow with WebP support"
else
    echo "Pillow was built without WebP; installing a binary wheel ..."
    # --force-reinstall, not --upgrade: a source-built Pillow can satisfy the
    # version range while still missing the encoder, and pip would then do
    # nothing ("already satisfied") and leave the venv broken.
    "${VENV_DIR}"/bin/pip install --force-reinstall --only-binary=:all: 'pillow>=10.4,<12.0'
    if ! pillow_has_webp; then
        echo "Error: Pillow still has no WebP support; the Gradio gallery will fail."
        echo "  Check the pip output above, or install libwebp-dev and rebuild Pillow."
        exit 1
    fi
    echo "Pillow with WebP support installed."
fi

# --- 3. OCR server env --------------------------------------------------------
# Upstream local_setup.sh installs paddlepaddle/paddleocr into deploy/fastapi/venv
# and then pre-downloads the CPU PP-OCRv5 models. That download is optional - the
# server fetches whatever is missing on first start - and it is the flaky part of
# the script, so a failure there must not abort the NPU setup below.
if [ -x "${FASTAPI_DIR}/venv/bin/python" ]; then
    echo "Already present: OCR server venv"
else
    echo "Setting up the OCR server (paddlepaddle, paddleocr) ..."
    (cd "${FASTAPI_DIR}" && ./local_setup.sh) || \
        echo "Warning: local_setup.sh reported an error; continuing and verifying the venv."
    if [ ! -x "${FASTAPI_DIR}/venv/bin/python" ]; then
        echo "Error: the OCR server venv was not created. Re-run ${FASTAPI_DIR}/local_setup.sh to see why."
        exit 1
    fi
    if ! "${FASTAPI_DIR}"/venv/bin/python -c 'import paddleocr' > /dev/null 2>&1; then
        echo "Error: paddleocr is missing from the OCR server venv."
        exit 1
    fi
    echo "OCR server venv verified (paddleocr importable)."
fi

# --- 4. NPU support -----------------------------------------------------------
# The OCR server only needs the dx_engine python binding; the DX-RT runtime
# itself is expected to be installed already (dxrt-cli -s works). Building
# dx_rt from source is what makes upstream local_deepx_setup.sh require sudo,
# and it is unnecessary here.
find_dx_rt() {
    local candidate
    for candidate in \
        "${DX_RT_PATH}" \
        "${HOME}/Desktop/deepx/SDK/dx-all-suite/dx-runtime/dx_rt" \
        "${HOME}/deepx/SDK/dx-all-suite/dx-runtime/dx_rt" \
        "${HOME}/dx-all-suite/dx-runtime/dx_rt" \
        "/opt/deepx/dx_rt"; do
        if [ -n "${candidate}" ] && [ -f "${candidate}/python_package/pyproject.toml" ]; then
            echo "${candidate}"
            return
        fi
    done
}

setup_npu() {
    if [ ! -f /usr/local/lib/libdxrt.so ] && [ ! -f /usr/lib/libdxrt.so ]; then
        echo "Skipping NPU setup: libdxrt.so not found (DX-RT is not installed)."
        echo "  Install DX-RT first (dx_rt/build.sh, needs sudo), then re-run this script."
        return
    fi

    local dx_rt
    dx_rt="$(find_dx_rt)"
    if [ -z "${dx_rt}" ]; then
        echo "Skipping NPU setup: no dx_rt checkout found. Pass --dx_rt PATH to enable it."
        return
    fi
    echo "Using dx_rt: ${dx_rt}"

    # Build in a copy: the cmake target drops _pydxrt.so back into its source tree
    if "${FASTAPI_DIR}"/venv/bin/python -c 'import dx_engine' > /dev/null 2>&1; then
        echo "Already present: dx_engine"
    else
        echo "Building dx_engine (python binding for DX-RT) ..."
        mkdir -p "${BUILD_DIR}"
        rm -rf "${BUILD_DIR}/dx_engine_src"
        cp -r "${dx_rt}/python_package" "${BUILD_DIR}/dx_engine_src"
        rm -f "${BUILD_DIR}"/dx_engine_src/src/dx_engine/capi/_pydxrt*.so
        # Without DX_ROOT_DIR the build cannot find lib/include and fails
        CMAKE_ARGS="-DDX_ROOT_DIR=${dx_rt}" \
            "${FASTAPI_DIR}"/venv/bin/pip install "${BUILD_DIR}/dx_engine_src"
    fi

    echo "Fetching the .dxnn NPU models ..."
    if ! (cd "${FASTAPI_DIR}" && ./setup_deepx_models.sh --deepx-path "${FASTAPI_DIR}/deepx"); then
        echo "Error: failed to fetch the NPU models."
        return 1
    fi

    # run.sh enables SETUP_NPU only when this file exists
    local env_file="${FASTAPI_DIR}/deepx_env.sh"
    local inter=1 intra=2 dynamic=1 max_load=3 in_workers=2 out_workers=4
    if [ -f "${FASTAPI_DIR}/.env.deepx" ]; then
        # shellcheck disable=SC1091
        source "${FASTAPI_DIR}/.env.deepx"
        inter="${CUSTOM_INTER_OP_THREADS_COUNT:-$inter}"
        intra="${CUSTOM_INTRA_OP_THREADS_COUNT:-$intra}"
        dynamic="${DXRT_DYNAMIC_CPU_THREAD:-$dynamic}"
        max_load="${DXRT_TASK_MAX_LOAD:-$max_load}"
        in_workers="${NFH_INPUT_WORKER_THREADS:-$in_workers}"
        out_workers="${NFH_OUTPUT_WORKER_THREADS:-$out_workers}"
    fi
    cat > "${env_file}" <<ENVEOF
#!/bin/bash
# DEEPX NPU environment for the OCR server. Written by build.sh; run.sh sources
# this file and enables SETUP_NPU when it exists. Values come from .env.deepx.
export CUSTOM_INTER_OP_THREADS_COUNT=${inter}
export CUSTOM_INTRA_OP_THREADS_COUNT=${intra}
export DXRT_DYNAMIC_CPU_THREAD=${dynamic}
export DXRT_TASK_MAX_LOAD=${max_load}
export NFH_INPUT_WORKER_THREADS=${in_workers}
export NFH_OUTPUT_WORKER_THREADS=${out_workers}
ENVEOF
    chmod +x "${env_file}"
    echo "Wrote deepx_env.sh (${inter} ${intra} ${dynamic} ${max_load} ${in_workers} ${out_workers})"

    # Fonts used when the server renders OCR overlays
    mkdir -p "${FASTAPI_DIR}/deepx/engine/fonts"
    cp "${SERVER_DIR}"/doc/fonts/*.ttf "${FASTAPI_DIR}/deepx/engine/fonts/" 2>/dev/null || true
}

if [ "$cpu_only" = true ]; then
    echo "Skipping NPU setup (--cpu-only). Removing deepx_env.sh so run.sh stays on CPU."
    rm -f "${FASTAPI_DIR}/deepx_env.sh"
else
    setup_npu
fi

# --- 5. Summary ---------------------------------------------------------------
echo
echo "Setup complete."
echo "  Web UI      : ${WEB_DIR}"
echo "  OCR server  : ${FASTAPI_DIR}"
if [ -f "${FASTAPI_DIR}/deepx_env.sh" ]; then
    echo "  Inference   : NPU (deepx_env.sh present)"
else
    echo "  Inference   : CPU only"
fi
echo
echo "Run the demo with ../../../scripts/run_ocr_web.sh (or the launcher card)."
