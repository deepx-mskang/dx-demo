#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="${ROOT_DIR}/apps/model-zoo"
BROWSER_PID_FILE="${APP_DIR}/.browser.pid"

# Load top-level configuration (DX_BROWSER, ...)
if [ -f "${ROOT_DIR}/config.sh" ]; then
    source "${ROOT_DIR}/config.sh"
fi

# The Model Zoo demo is a static HTML page - there is no apps/model-zoo/cpp or
# apps/model-zoo/python. Both backends therefore do the same thing: open the
# page. DX_BACKEND only picks a backend where one exists.
if [ "${DX_BACKEND:-}" == "python" ]; then
    echo "Model Zoo is a static HTML page; the Python backend opens the same file."
fi

# DX_ModelZoo_latest.html is the symlink apps/model-zoo/update_modelzoo.sh
# repoints at each freshly published page; DX_MODELZOO_HTML overrides it. The
# older retouched pages act as a fallback, so the demo still starts on a machine
# that has never run update_modelzoo.sh.
HTML="${DX_MODELZOO_HTML:-${APP_DIR}/DX_ModelZoo_latest.html}"
if [ ! -f "${HTML}" ]; then
    HTML="$(ls -1t "${APP_DIR}"/*.html 2>/dev/null | head -n 1)"
fi

if [ -z "${HTML}" ] || [ ! -f "${HTML}" ]; then
    echo "Error: no Model Zoo page in ${APP_DIR}. Run ./apps/model-zoo/update_modelzoo.sh"
    read -t 3 -p "Press enter to exit..." || true
    exit 1
fi

# Close a page left open by a previous Start, so repeated clicks do not stack
# browser windows the Stop button can no longer reach (it only tracks one PID).
"${ROOT_DIR}"/scripts/kill_modelzoo.sh

echo "Opening $(basename "${HTML}")"
# open_browser.sh resolves the desktop's default browser, turns the path into a
# file:// URL (a bare filename is treated as a search term by most browsers) and
# execs the browser, so $! is the browser itself. DX_BROWSER overrides the choice.
#
# --profile-dir is what makes Stop reliable: on its own profile the browser is
# always a process we own. Without it, a Firefox or Chrome the user already had
# open swallows the URL and exits our process, leaving nothing to stop.
"${ROOT_DIR}"/scripts/open_browser.sh --profile-dir "${APP_DIR}/browser-profile" "${HTML}" &
BROWSER_PID=$!
echo "${BROWSER_PID}" > "${BROWSER_PID_FILE}"

# Let the launcher end its "Wait" once the browser process is really up, instead
# of holding the card's buttons for the card's full loading_sec.
if [ -n "${DX_LAUNCHER_READY_FILE:-}" ]; then
    # open_browser.sh execs the browser, so the PID stops being a shell once the
    # browser has taken over; give the window a moment to paint after that.
    for _ in $(seq 1 40); do
        comm="$(cat "/proc/${BROWSER_PID}/comm" 2>/dev/null)" || break
        case "${comm}" in
            ""|bash|sh|open_browser.sh) sleep 0.25;;
            *) break;;
        esac
    done
    sleep 2
    : > "${DX_LAUNCHER_READY_FILE}"
fi

wait "${BROWSER_PID}"

# The browser was closed by hand rather than by Stop: drop the stale PID, which
# the kernel may hand to an unrelated process later.
if [ -f "${BROWSER_PID_FILE}" ] && [ "$(cat "${BROWSER_PID_FILE}")" == "${BROWSER_PID}" ]; then
    rm -f "${BROWSER_PID_FILE}"
fi
