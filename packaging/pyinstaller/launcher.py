"""Entry point of the standalone builds: the desktop application.

PyInstaller freezes this file with the package and its dependencies into
one folder per operating system; double-clicking the executable inside it
starts the local server and opens the browser. ``--self-test`` runs a
simulated analysis and exits, which is how the build workflow checks that
a freshly frozen bundle actually works.
"""

import multiprocessing
import sys

from astrovision.gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()          # harmless elsewhere, required on Windows
    sys.exit(main())
