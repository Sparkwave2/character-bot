#!/usr/bin/env bash
# Installs Character Bot on Linux:
#   1. system packages it needs (Python 3.10+ with Tk, Opus for voice chat, xclip for pasting pictures)
#   2. a Python environment in .venv with the app's requirements
#   3. optional extras: Whisper on the GPU, and the local Kokoro voice
#   4. optionally a menu entry, so it shows up with your other apps
#
# Usage: ./install.sh            asks before each step
#        ./install.sh --yes      accepts everything (GPU extras only if an NVIDIA GPU is found)
#        ./install.sh --no-system-packages   skips step 1 (no sudo)
# Run it again at any time to update the Python packages; models and llama.cpp download from the app.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

YES=0
SYSTEM=1
for arg in "$@"; do
    case "$arg" in
        -y|--yes) YES=1 ;;
        --no-system-packages) SYSTEM=0 ;;
        -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $arg (see --help)" >&2; exit 1 ;;
    esac
done

bold=$(tput bold 2>/dev/null || true); dim=$(tput dim 2>/dev/null || true); reset=$(tput sgr0 2>/dev/null || true)
step() { echo; echo "${bold}==> $*${reset}"; }
note() { echo "${dim}    $*${reset}"; }
ask() {   # ask "question" default(y/n)
    local prompt="[Y/n]" reply
    [ "$2" = n ] && prompt="[y/N]"
    if [ "$YES" = 1 ] || [ ! -t 0 ]; then [ "$2" = y ]; return; fi
    read -r -p "    $1 $prompt " reply
    reply=${reply:-$2}
    [[ "$reply" =~ ^[Yy] ]]
}

if [ "$(id -u)" = 0 ]; then SUDO=""; else SUDO="sudo"; fi
HAS_NVIDIA=0
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 && HAS_NVIDIA=1

# ---- 1. system packages ------------------------------------------------------
if [ "$SYSTEM" = 1 ]; then
    step "System packages"
    if command -v apt-get >/dev/null; then
        PM="apt"; PKGS="python3 python3-venv python3-pip python3-tk libopus0 xclip"
        INSTALL="$SUDO apt-get install -y"; REFRESH="$SUDO apt-get update"
    elif command -v dnf >/dev/null; then
        PM="dnf"; PKGS="python3 python3-pip python3-tkinter opus xclip"
        INSTALL="$SUDO dnf install -y"; REFRESH=""
    elif command -v pacman >/dev/null; then
        PM="pacman"; PKGS="python python-pip tk opus xclip"
        INSTALL="$SUDO pacman -S --needed --noconfirm"; REFRESH=""
    elif command -v zypper >/dev/null; then
        PM="zypper"; PKGS="python3 python3-pip python3-tk libopus0 xclip"
        INSTALL="$SUDO zypper --non-interactive install"; REFRESH=""
    else
        PM=""
        note "Couldn't recognise your package manager. Install these yourself if they're missing:"
        note "Python 3.10+ with venv and Tkinter, the Opus library, and xclip."
    fi
    if [ -n "$PM" ]; then
        note "Needs ($PM): $PKGS"
        if ask "Install them now${SUDO:+ (uses sudo)}?" y; then
            [ -n "$REFRESH" ] && $REFRESH
            # shellcheck disable=SC2086
            $INSTALL $PKGS
        else
            note "Skipped."
        fi
    fi
fi

# ---- 2. Python environment --------------------------------------------------
step "Python environment"
PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null && "$cand" - <<'EOF' 2>/dev/null
import sys, tkinter, venv
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF
    then PY=$(command -v "$cand"); break; fi
done
if [ -z "$PY" ]; then
    echo "No Python 3.10 or newer with Tkinter was found." >&2
    echo "Install Python 3.10+ together with its Tk package (python3-tk / python3-tkinter / tk), then run this again." >&2
    exit 1
fi
note "Using $PY ($("$PY" --version))"
if [ -x .venv/bin/python ] && ! .venv/bin/python -c "import tkinter" 2>/dev/null; then
    note "The existing .venv can't use Tkinter; making a new one."
    rm -rf .venv
fi
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade --quiet pip
note "Installing the requirements (this takes a few minutes the first time)…"
.venv/bin/python -m pip install --upgrade -r requirements.txt
cp requirements.txt .venv/installed-requirements.txt

# ---- 3. optional extras -------------------------------------------------------
step "Optional extras"
if [ "$HAS_NVIDIA" = 1 ]; then
    note "NVIDIA GPU found: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
    if ask "Install the GPU libraries for speech recognition (Whisper, ~1 GB)?" y; then
        .venv/bin/python -m pip install --upgrade nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
    fi
else
    note "No NVIDIA GPU found: speech recognition will run on the CPU."
fi
if ask "Install the local voice (Kokoro: free text-to-speech on this PC instead of ElevenLabs)?" "$([ "$HAS_NVIDIA" = 1 ] && echo y || echo n)"; then
    if [ "$HAS_NVIDIA" = 1 ]; then
        .venv/bin/python -m pip install --upgrade onnxruntime-gpu espeakng-loader phonemizer \
            nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
    else
        .venv/bin/python -m pip install --upgrade onnxruntime espeakng-loader phonemizer
    fi
    # --no-deps: kokoro-onnx would otherwise swap the GPU build of onnxruntime for the CPU one
    .venv/bin/python -m pip install --upgrade --no-deps kokoro-onnx
    note "Turn it on under Voice → Use the local voice. The voice model (~350 MB) downloads on the first /join."
fi

# ---- 4. launcher --------------------------------------------------------------
chmod +x run.sh
step "Menu entry"
DESKTOP="${XDG_DATA_HOME:-$HOME/.local/share}/applications/character-bot.desktop"
if ask "Add Character Bot to your applications menu?" y; then
    mkdir -p "$(dirname "$DESKTOP")"
    cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=Character Bot
Comment=Local AI characters for Discord and a built-in chat
Exec="$PWD/run.sh"
Path=$PWD
Terminal=false
Categories=Network;Chat;
EOF
    note "Added $DESKTOP"
fi

step "Done"
echo "    Start it with:  $PWD/run.sh"
echo "    Then: Model → Choose a model… to download one, and Load. llama.cpp installs itself on the first Load."
