# The desktop application

`astrovision gui` opens AstroVision-X as a desktop application: a window
in your browser, served by the package itself on your own machine. It does
what the command line does -- analyse an image or a series, simulate a
field, read alerts, hand candidates to the vetting page -- with a file
browser, a progress bar per pipeline stage, and the report, the catalog and
every cutout a click away.

![The analysis screen](../figures/gui_summary.png)

It needs no GUI toolkit. The server is Python's standard library, the page
is one HTML file, and the browser is whatever the PC already has. That is
why it runs anywhere Python runs, and why it can be frozen into a folder
that runs where Python does not.

## Installing on any PC

Three ways, from the least to the most self-contained.

**With Python already installed** (any OS, Python 3.9 or newer):

```bash
pip install "astrovision-x[science,ml] @ git+https://github.com/thecodebasedot/AstroVision-X.git"
astrovision-gui
```

**One-click installers** that create a private Python environment for the
application, install it there, and put a launcher where you expect one:

| OS | Do this | You get |
| --- | --- | --- |
| Windows | double-click `packaging\install.bat` (or run `install.ps1` in PowerShell) | `AstroVision-X` on the Desktop and in the Start Menu; everything in `%LOCALAPPDATA%\AstroVision-X` |
| macOS | `bash packaging/install.sh` | `~/Applications/AstroVision-X.command` to double-click, and `astrovision-gui` in `~/.local/bin`; everything in `~/.astrovision-x` |
| Linux | `bash packaging/install.sh` | an AstroVision-X entry in the applications menu and `astrovision-gui` in `~/.local/bin`; everything in `~/.astrovision-x` |

Both scripts need Python 3.9+ on the PATH and say so if it is missing.
Run from a clone they install that clone; run on their own they install
from GitHub. To uninstall, delete the folder they name and the shortcuts.

**Standalone builds, no Python needed.** The `Desktop builds` workflow
(`.github/workflows/build.yml`) freezes the application with PyInstaller
for Windows, macOS and Linux and attaches the archives to every tagged
release. Unpack one and start `AstroVision-X` (`AstroVision-X.exe` on
Windows) inside the folder; a terminal shows the address and the log, and
Ctrl-C there stops it. Each build runs `--self-test`, a simulated field
through the whole pipeline, before it is published, so a build that exists
is one that ran. The standalone builds carry the science stack (SciPy,
astropy, scikit-image, scikit-learn) but not PyTorch, which is a gigabyte;
the deep-learning extras remain available in a pip installation. To build
one yourself:

```bash
pip install -e ".[science]" pyinstaller
pyinstaller packaging/pyinstaller/astrovision-gui.spec
./dist/AstroVision-X/AstroVision-X --self-test
```

## The screens

**Analyse an image.** Pick a FITS (or npy/png) file in the browser on the
left; the header is read at once -- size, band, epoch, pixel scale, the
frame centre in ICRS, and a note when the WCS had to be refitted from
another frame -- and a preview is shown. Choose a preset, threshold,
redshift, output folder and reports, optionally a catalog database, and
run. The right-hand pane lists every stage as it starts and finishes, with
its time, above the pipeline's log. When it finishes:

- *summary*: counts by class, transients, lens candidates, anomalies; the
  warnings the run raised (an undersampled PSF, a missing zero point, a
  failed stage) in the place they are least easy to miss; a button to open
  the vetting page for the ranked candidates.
- *candidates*: the ranked follow-up list with a cutout, the score and
  verdict, the reasons and the caveats, exactly as the report gives them.
- *catalog*: every source, sortable by any column, filterable by text, a
  click on a row showing its cutout.
- *report*: the HTML report itself.
- *image*: the whole frame, asinh-stretched, north up.
- *files*: what was written and where.

**Series & transients.** Select two or more epochs of the same field; they
are aligned, PSF-matched and differenced, and transients are scored
real/bogus. The results have the same tabs, and the summary lists the
transients; an Avro alert file can be written in the same run.

**Simulate a field.** Stars, galaxies, nebulae, clusters, lenses,
anomalies, size, seed; one image or a series with transients. The truth
table is written beside the image and an "Analyse this image" button
takes it straight to the analysis screen -- the quickest way to see what
the pipeline does before pointing it at real data.

**Alerts.** Pick an Avro file (this package's, ZTF's or Rubin's); the
packets are listed as received, and "Open the vetting page" shows each
with its cutouts, light curve and scores.

**Runs.** Everything run in this session, with its status and time;
clicking one brings its results back.

**About.** Version, Python, which optional backends are installed, and the
boundary this software keeps: it ranks candidates, and never declares a
discovery.

## What it is, technically

`astrovision.gui.app` is a `ThreadingHTTPServer` on 127.0.0.1 with a JSON
API (`/api/status`, `/api/browse`, `/api/inspect`, `/api/analyze`,
`/api/series`, `/api/simulate`, `/api/alerts`, `/api/vet`, `/api/jobs/…`
with `report.html`, `report.json`, `catalog`, `candidates`, `cutout.png`,
`preview.png`); `astrovision.gui.page` is the page. Each run is a job on
its own thread; the pipeline reports each stage through the `progress`
callback `Pipeline` now accepts, and the package's log lines are routed
to the job that produced them. The tests in `tests/test_gui.py` drive the
whole API the way the page does: simulate, analyse, read every result,
hand off to vetting.

The server can read any file the user can. That is what a desktop
application is for and why it binds to localhost only; never expose it on
a network. `--host`, `--port`, `--workdir` and `--no-browser` exist for the
odd case (a remote desktop, a second instance), not for that.
