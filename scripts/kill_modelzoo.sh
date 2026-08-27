#!/usr/bin/env bash
set -euo pipefail

# Close the Model Zoo page opened by run_modelzoo.sh.
#
# The browser is killed by the PID run_modelzoo.sh recorded at launch, not by a
# command-line pattern: open_browser.sh picks the desktop's default browser, so
# the process may be firefox, chromium, chrome or brave, and the old blanket
# 'pkill -f chrome' both missed firefox and closed every unrelated Chrome window
# the user had open.
#
# Fallback, for a PID file lost across a reboot or a page opened by hand: match
# the browser command line, which carries the file:// URL of the page. This is
# specific enough not to hit unrelated windows.
#
# Not covered: if the page was opened as a tab in an ALREADY running Chrome or
# Chromium, the launched process hands the URL over and exits immediately, so
# there is no process to kill - close that tab by hand (Ctrl+W), or with Alt+F4
# for a fullscreen window.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="${ROOT_DIR}/apps/model-zoo"
BROWSER_PID_FILE="${APP_DIR}/.browser.pid"

killed=false

if [ -f "${BROWSER_PID_FILE}" ]; then
    pid="$(cat "${BROWSER_PID_FILE}" 2>/dev/null || true)"
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true
        # SIGTERM lets the browser shut down cleanly. SIGKILL is a last resort:
        # Firefox counts a hard kill as a startup crash, and after a few of them
        # it opens a "Troubleshoot Mode?" dialog instead of the page - which is
        # exactly what a demo must not do. Give it 10s before escalating.
        for _ in $(seq 1 40); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 0.25
        done
        kill -KILL "${pid}" 2>/dev/null || true
        killed=true
    fi
    rm -f "${BROWSER_PID_FILE}"
fi

if [ "${killed}" != "true" ]; then
    # Matches '<browser> ... file:///.../apps/model-zoo/<page>.html'. The
    # file:// prefix keeps this off editors or shells that merely mention the
    # path - open_browser.sh is what turns the path into a URL.
    pkill -TERM -f "file://[^ ]*apps/model-zoo/[^ ]*\.html" 2>/dev/null || true
fi

# run_modelzoo.sh is deliberately NOT killed here: it waits on the browser and
# exits by itself, and it calls this script before opening the page - a pattern
# kill would abort the very start that invoked it.

exit 0
