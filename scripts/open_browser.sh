#!/bin/bash

# Open a URL (or a local HTML file) in the desktop's default browser.
# Adds a kiosk-style fullscreen flag when the browser is known to support one.
#
# Usage: open_browser.sh [--no-fullscreen] [--profile-dir <dir>] <url-or-file>
#
# --profile-dir runs the browser on its own profile, as its own process. Without
# it, a browser that is ALREADY running takes the URL over and the process we
# started exits at once - which leaves the caller with no PID to stop later, and
# puts the demo page in the middle of whatever the user was browsing.
#
# Set DX_BROWSER in config.sh to force a specific browser command.

FULLSCREEN=true
PROFILE_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --no-fullscreen) FULLSCREEN=false; shift;;
        --profile-dir) PROFILE_DIR="${2:-}"; shift 2;;
        *) break;;
    esac
done

TARGET="$1"

if [ -z "${TARGET}" ]; then
    echo "Usage: $(basename "$0") [--no-fullscreen] [--profile-dir <dir>] <url-or-file>"
    exit 1
fi

# Local file (e.g. the Model Zoo HTML) -> file:// URL
if [ -f "${TARGET}" ]; then
    TARGET="file://$(realpath "${TARGET}")"
fi

# Resolve the desktop's default browser, e.g. "firefox_firefox.desktop" -> "firefox"
default_browser() {
    local desktop
    desktop="$(xdg-settings get default-web-browser 2>/dev/null)"
    if [ -z "${desktop}" ]; then
        desktop="$(xdg-mime query default text/html 2>/dev/null | head -n 1)"
    fi
    desktop="${desktop%.desktop}"
    echo "${desktop%%_*}"
}

# Flag that opens the browser fullscreen, empty when unknown or not wanted
fullscreen_flag() {
    if [ "${FULLSCREEN}" != "true" ]; then
        echo ""
        return
    fi
    case "$1" in
        chromium*|*chrome*|brave*) echo "--start-fullscreen";;
        firefox*) echo "--kiosk";;
        *) echo "";;
    esac
}

# Flags that keep this window out of an already running browser, empty when no
# profile directory was asked for or the browser is unknown
profile_flags() {
    if [ -z "${PROFILE_DIR}" ]; then
        return
    fi
    # Firefox refuses a --profile path that does not exist yet; Chromium creates
    # its own. Snap-confined Firefox can only reach non-hidden paths under $HOME,
    # so callers should keep the directory out of a dot-directory.
    mkdir -p "${PROFILE_DIR}"
    case "$1" in
        chromium*|*chrome*|brave*)
            printf '%s' "--user-data-dir=${PROFILE_DIR}"
            ;;
        firefox*)
            # A profile of our own also keeps Firefox's crash counter away from
            # the user's: a stopped demo must never turn into a "Troubleshoot
            # Mode?" prompt on the next start.
            if [ ! -f "${PROFILE_DIR}/user.js" ]; then
                cat > "${PROFILE_DIR}/user.js" <<'PREFS'
user_pref("toolkit.startup.max_resumed_crashes", -1);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.shell.checkDefaultBrowser", false);
PREFS
            fi
            printf '%s' "--no-remote --profile ${PROFILE_DIR}"
            ;;
    esac
}

BROWSER="${DX_BROWSER:-$(default_browser)}"

if [ -n "${BROWSER}" ] && command -v "${BROWSER}" > /dev/null 2>&1; then
    echo "Opening ${TARGET} in ${BROWSER}"
    exec "${BROWSER}" $(profile_flags "${BROWSER}") $(fullscreen_flag "${BROWSER}") "${TARGET}"
fi

# Unknown browser: hand the URL over to the desktop handler (no fullscreen)
for opener in xdg-open sensible-browser x-www-browser; do
    if command -v "${opener}" > /dev/null 2>&1; then
        echo "Opening ${TARGET} with ${opener}"
        exec "${opener}" "${TARGET}"
    fi
done

echo "Error: no browser found. Open ${TARGET} manually."
exit 1
