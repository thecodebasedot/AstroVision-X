#!/usr/bin/env bash
# Install AstroVision-X for the current user on Linux or macOS.
#
#   bash packaging/install.sh            # from a clone of the repository
#   curl -fsSL <raw url of this file> | bash    # from anywhere
#
# Needs Python 3.9 or newer on the PATH.  Everything goes into one folder,
# ~/.astrovision-x, and a launcher is written to ~/.local/bin/astrovision-gui
# (plus a desktop entry on Linux).  Remove that folder to uninstall.
set -euo pipefail

PREFIX="${ASTROVISION_HOME:-$HOME/.astrovision-x}"
SOURCE="${ASTROVISION_SOURCE:-}"          # a local clone; empty = from GitHub
REPO="${ASTROVISION_REPO:-https://github.com/thecodebasedot/AstroVision-X.git}"
EXTRAS="${ASTROVISION_EXTRAS:-science,ml}"

find_python() {
  for candidate in python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        echo "$candidate"; return 0
      fi
    fi
  done
  return 1
}

PYTHON="$(find_python)" || { echo "AstroVision-X needs Python 3.9 or newer. Install it from https://www.python.org/downloads/ and run this again." >&2; exit 1; }
echo "Using $("$PYTHON" --version) at $(command -v "$PYTHON")"

if [ -z "$SOURCE" ]; then
  here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
  if [ -n "$here" ] && [ -f "$here/../pyproject.toml" ]; then
    SOURCE="$(cd "$here/.." && pwd)"
  fi
fi

mkdir -p "$PREFIX"
if [ ! -x "$PREFIX/venv/bin/python" ]; then
  echo "Creating a private Python environment in $PREFIX/venv"
  "$PYTHON" -m venv "$PREFIX/venv"
fi
VPY="$PREFIX/venv/bin/python"
"$VPY" -m pip install --quiet --upgrade pip

if [ -n "$SOURCE" ]; then
  echo "Installing from $SOURCE with extras [$EXTRAS]"
  "$VPY" -m pip install --quiet --upgrade "$SOURCE[$EXTRAS]"
else
  echo "Installing from $REPO with extras [$EXTRAS]"
  "$VPY" -m pip install --quiet --upgrade "astrovision-x[$EXTRAS] @ git+$REPO"
fi

mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/astrovision-gui" <<LAUNCH
#!/usr/bin/env bash
exec "$VPY" -m astrovision.gui "\$@"
LAUNCH
cat > "$HOME/.local/bin/astrovision" <<LAUNCH
#!/usr/bin/env bash
exec "$VPY" -m astrovision.cli.main "\$@"
LAUNCH
chmod +x "$HOME/.local/bin/astrovision-gui" "$HOME/.local/bin/astrovision"

if [ "$(uname -s)" = "Linux" ] && [ -d "$HOME/.local/share" ]; then
  mkdir -p "$HOME/.local/share/applications"
  cat > "$HOME/.local/share/applications/astrovision-x.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=AstroVision-X
Comment=Computer vision and machine learning for astronomical images
Exec=$HOME/.local/bin/astrovision-gui
Terminal=true
Categories=Science;Education;
DESKTOP
fi
if [ "$(uname -s)" = "Darwin" ]; then
  APP="$HOME/Applications/AstroVision-X.command"
  mkdir -p "$HOME/Applications"
  printf '#!/usr/bin/env bash\nexec "%s" -m astrovision.gui\n' "$VPY" > "$APP"
  chmod +x "$APP"
  echo "A double-clickable launcher is at $APP"
fi

"$VPY" -m astrovision.cli.main info | tail -n +9
echo
echo "Installed. Start the desktop application with:  astrovision-gui"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "(add $HOME/.local/bin to your PATH, or run $HOME/.local/bin/astrovision-gui)";; esac
