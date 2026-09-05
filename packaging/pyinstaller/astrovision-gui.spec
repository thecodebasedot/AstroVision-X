# -*- mode: python ; coding: utf-8 -*-
# PyInstaller recipe for the AstroVision-X desktop application.
#
#   pyinstaller packaging/pyinstaller/astrovision-gui.spec
#
# builds dist/AstroVision-X/ -- one folder holding the executable, Python
# and every library it needs.  A folder rather than a single file because
# NumPy, SciPy and astropy unpack far faster from disk than from a
# self-extracting archive, and because antivirus software is happier.

import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
here = os.path.dirname(os.path.abspath(SPEC))
root = os.path.abspath(os.path.join(here, "..", ".."))

hidden = collect_submodules("astrovision")
datas = []
# Optional science stack: bundled when installed, absent otherwise.  The
# application says which backends it has on its About page.
for package in ("scipy", "astropy", "skimage", "sklearn"):
    try:
        __import__(package)
    except ImportError:
        continue
    hidden += collect_submodules(package)
    datas += collect_data_files(package)

a = Analysis(
    [os.path.join(here, "launcher.py")],
    pathex=[root],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    # PyTorch is a gigabyte and is not part of the desktop build; the ml
    # extras still work in a pip installation.
    # astropy imports its own astropy.tests.runner at start-up, so that one
    # stays in.
    excludes=["torch", "torchvision", "tkinter", "matplotlib.tests", "scipy.tests",
              "numpy.tests", "sklearn.tests", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AstroVision-X",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,          # the terminal shows the URL and the log; Ctrl-C stops it
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="AstroVision-X",
)
