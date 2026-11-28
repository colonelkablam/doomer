#!/usr/bin/env bash
# OK Doomer for Linux: run this (./play.sh), or pick OK Doomer from the
# applications menu once this has made its entry. The Linux twin of
# play.bat: the first run sets up Python's environment (a minute or two);
# every run updates the game first, from the branch this folder is on, so
# it matches everyone else's (different versions can't play together),
# then starts it. Started from the menu (no terminal), what it prints goes
# to ~/.ok-doomer/play.log, and a failure pops up a notification.
set -u

cd "$(dirname "$(readlink -f "$0")")" || exit 1
here="$(pwd)"

if [ ! -t 1 ]; then
    mkdir -p "$HOME/.ok-doomer"
    exec >>"$HOME/.ok-doomer/play.log" 2>&1
    echo "--- $(date)"
fi

fail() {
    echo
    echo "$1"
    if [ -t 0 ]; then
        read -rp "Press Enter to close. " _
    elif command -v notify-send >/dev/null; then
        notify-send "OK Doomer" "$1 (more in ~/.ok-doomer/play.log)"
    fi
    exit 1
}

if [ ! -x .venv/bin/python ]; then
    echo "First run: setting up, this takes a minute or two..."
    python3 -m venv .venv || fail "Couldn't set up: is Python 3.11 or newer installed? \
(Debian, Ubuntu, Raspberry Pi OS also need: sudo apt install python3-venv)"
fi

echo "Checking for updates..."
git pull --ff-only || echo "(Couldn't update: playing the version you have.)"

# Once: an "OK Doomer" entry in the applications menu, with the game's icon.
# The marker stops it coming back if you delete the entry; delete
# .shortcut-made to have it made again.
if [ ! -e .shortcut-made ]; then
    apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$apps"
    cat >"$apps/ok-doomer.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=OK Doomer
Comment=A Doom-style LAN shooter in a 1980s office
Exec="$here/play.sh"
Path=$here
Icon=$here/ok-doomer.png
Terminal=false
Categories=Game;ActionGame;
EOF
    echo "Added OK Doomer to the applications menu."
    echo made >.shortcut-made
fi

.venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements.txt \
    || fail "Couldn't install what the game needs: see above."

.venv/bin/python -m ok_doomer.client "$@" || fail "The game stopped with an error: see above."
