#!/usr/bin/env bash
# Starts Character Bot (run ./install.sh once first).
cd "$(dirname "$(readlink -f "$0")")" || exit 1
if [ ! -x .venv/bin/python ]; then
    echo "Character Bot isn't installed yet. Run ./install.sh first." >&2
    exit 1
fi
# after an update (git pull), install whatever requirements changed
if ! cmp -s requirements.txt .venv/installed-requirements.txt; then
    .venv/bin/python -m pip install -r requirements.txt && cp requirements.txt .venv/installed-requirements.txt
fi
# The GPU libraries for speech (pip's nvidia-* packages) are found through LD_LIBRARY_PATH on Linux.
nvidia_libs=$(.venv/bin/python - <<'EOF'
import glob, os, sys
print(os.pathsep.join(sorted(glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "nvidia", "*", "lib")))))
EOF
)
if [ -n "$nvidia_libs" ]; then
    export LD_LIBRARY_PATH="$nvidia_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec .venv/bin/python main.py "$@"
