#!/bin/sh
# One-step setup of this fork for Qwen3.6-35B-A3B (Unsloth UD-Q4_K_M with its MTP block) on Linux. Nothing of the model is downloaded:
# the first run asks where its files are (or pass the path: fast-setup-35b.sh /path/to/model), builds the engine, then tunes.
# Later runs only tune. tools/fast_setup_q36.py does the work; --help lists its options.
cd "$(dirname "$0")" || exit 1
# Python and the .venv first (setup.sh installs Python through apt/dnf/pacman if it is missing); nothing else is set up here
sh ./setup.sh --python-only || exit 1
PY=.venv/bin/python
[ -x python/bin/python3 ] && PY=python/bin/python3
export PYTHONUTF8=1
exec "$PY" tools/fast_setup_q36.py "$@"
