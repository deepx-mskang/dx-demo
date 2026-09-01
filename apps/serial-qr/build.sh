#!/bin/bash
# serial-qr 데모 빌드: C++ 서버 + 웹 프론트엔드
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

clean_build=false
skip_web=false

while (( $# )); do
    case "$1" in
        --clean) clean_build=true; shift;;
        --no-web) skip_web=true; shift;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--clean] [--no-web]"; exit 1;;
    esac
done

# ---------------------------------------------------------------------------
# 1) C++ 서버
# ---------------------------------------------------------------------------
echo "[serial-qr] building C++ server..."
if [ "$clean_build" = true ]; then
    "${SCRIPT_DIR}/cpp/build.sh" --clean
else
    "${SCRIPT_DIR}/cpp/build.sh"
fi

if [ "$skip_web" = true ]; then
    echo "[serial-qr] --no-web: 프론트엔드 빌드를 건너뜁니다."
    exit 0
fi

# ---------------------------------------------------------------------------
# 2) 웹 프론트엔드 (Vite + React)
# ---------------------------------------------------------------------------

# nvm 로 설치한 node 는 로그인 셸에만 있으므로 여기서 한 번 더 로드한다.
if ! command -v npm >/dev/null 2>&1 && [ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]; then
    # shellcheck disable=SC1091
    . "${NVM_DIR:-$HOME/.nvm}/nvm.sh"
fi

if ! command -v npm >/dev/null 2>&1; then
    cat >&2 <<'MSG'

[serial-qr] ERROR: npm 을 찾을 수 없습니다.

  이 데모의 프론트엔드는 Vite + React 로 만들어져 Node.js 20 이상이 필요합니다.
  nvm 으로 설치하는 것을 권장합니다 (sudo 불필요):

    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    source ~/.nvm/nvm.sh && nvm install 20

  설치 후 이 스크립트를 다시 실행하세요.
  (C++ 서버만 빌드하려면 --no-web 을 쓰세요.)

MSG
    exit 1
fi

echo "[serial-qr] building web frontend with $(node -v)..."
cd "${SCRIPT_DIR}/web"

if [ "$clean_build" = true ]; then
    rm -rf node_modules dist
fi

if [ -f package-lock.json ]; then
    npm ci
else
    npm install
fi

npm run build

echo "[serial-qr] build complete."
echo "[serial-qr] 실행: scripts/run_serial_qr.sh"
