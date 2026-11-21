# Code for analysing RIXS spectra from VERITAS

## Layout

| Folder | Contents |
|---|---|
| `src/` | Shared analysis code, installed as top-level modules (see [Using the code](#using-the-code)) |
| `analysis_notebooks/<your-name>/` | Your own notebooks. Don't edit someone else's notebook: copy it into your own folder. |
| `energy_scales/` | Energy calibrations (`.npy`). **Never overwrite one.** `energy_scales/index.toml` says which runs use which file |
| `runinfo_files/` | Run tables of older beamtimes (read-only `.xlsx`/`.csv`) and the n26 run log CSV fetched from Google Sheets |
| `tools/` | Scripts: `fetch_runlog.py` (n26 run log, see [docs/runlog_sync.md](docs/runlog_sync.md)), `check_run_mapping.py` |
| `tests/` | Checks that need no raw data: `uv run pytest` |

Raw data (`.h5`) is **not** stored in the repository.

## One-time setup

```bash
git clone https://github.com/robintyb/VERITAS_ANALYSIS.git
cd VERITAS_ANALYSIS
python -m venv .venv && source .venv/bin/activate   # or a conda env
pip install -e ".[dev]"          # installs src/ in editable mode + tools
pre-commit install               # strips notebook outputs on every commit
cp config.example.toml config.local.toml   # then edit your raw-data paths
```

`rixsaqlib` (the VERITAS DLD reader) is not on PyPI. A copy is vendored in
`extern/rixsaqlib/` and gets installed by the `pip install -e` step above.

### With uv

If you use [uv](https://docs.astral.sh/uv/), skip the venv and pip lines:

```bash
uv sync --extra dev              # creates .venv, installs uv.lock versions, src/ + rixsaqlib editable
uv run pre-commit install
cp config.example.toml config.local.toml
```

`dev` is an extra, so pass `--extra dev`, not `--group dev`. A plain `uv sync`
removes the dev tools.

`fast` is an optional extra (numba). It compiles the `droplet_temperature` simulation loop,
which takes a default run from about 11 s to under a second with identical results:
`uv sync --extra dev --extra fast`. Without it everything still works, in numpy. Any `uv sync` that
leaves `--extra fast` out removes numba again, so repeat it each time.

Run `uv sync` in a terminal on the machine where your Jupyter kernel runs
(on JupyterHub: **File → New → Terminal**). The install records absolute
paths, and on some nodes `/data/visitors` is a link to another path. A
`.venv` synced on another node can then give
`ModuleNotFoundError: No module named 'RIXS_functions'` in the notebook. Run `uv sync --extra dev` again after pulling
dependency changes. Either `source .venv/bin/activate` or prefix commands
with `uv run` (e.g. `uv run jupyter lab`).

## Jupyter kernel

Register the project's `.venv` as a named Jupyter kernel once. The kernel
then shows up in JupyterLab, JupyterHub and VS Code even if Jupyter itself
runs from another environment (e.g. the facility Anaconda). It runs
`.venv/bin/python`, and `src/` is installed in editable mode, so the kernel
always sees the current source and whatever is installed in `.venv`. You do
not need to re-register it when code or packages change.

With pip / venv:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name veritas-analysis --display-name "Python (VERITAS analysis)"
```

With uv:

```bash
uv run python -m ipykernel install --user --name veritas-analysis --display-name "Python (VERITAS analysis)"
```

`ipykernel` comes from the `dev` extra, so do the one-time setup above first.
Then pick **Python (VERITAS analysis)** as the kernel in your notebook.
`jupyter kernelspec list` shows the registered kernels;
`jupyter kernelspec uninstall veritas-analysis` removes this one.

The kernel stores the absolute path to `.venv/bin/python`. Rebuilding `.venv`
in the same place (`uv sync`, or deleting it and re-running the pip setup)
keeps the kernel working. If you move or re-clone the repo, run the install
command again (the same `--name` overwrites the old entry). If you have more
than one clone, give each its own `--name` and `--display-name`.

Start every notebook with:

```python
%load_ext autoreload
%autoreload 2
import RIXS_functions as rf
import fitting_functions as ff
```

and add the modules you use from [Using the code](#using-the-code), e.g.
`import daq` and `import spectrum as sp` for n26 data.

### Keeping the kernel in sync

| You changed... | pip / venv | uv |
|---|---|---|
| A function in an existing `src/` module (edit, add, or delete) | Nothing. `%autoreload 2` reloads the module before the next cell runs | same |
| A new module file in `src/` | Add it to `py-modules` in `pyproject.toml`, run `pip install -e ".[dev]"`, restart the kernel | Add it to `py-modules`, run `uv sync --extra dev`, restart the kernel |
| Added a package | Add it to `dependencies` in `pyproject.toml`, run `pip install -e ".[dev]"`, restart the kernel | `uv add <package>`, restart the kernel |
| Removed a package | Remove it from `dependencies`, `pip uninstall <package>`, restart the kernel | `uv remove <package>`, restart the kernel |
| Pulled someone else's dependency changes from `main` | `pip install -e ".[dev]"`, restart the kernel | `uv sync --extra dev`, restart the kernel |

Notes:

- Autoreload also updates edited functions you imported directly with
  `from RIXS_functions import <name>`. Under `%autoreload 2`, two kinds of
  directly imported names go stale: a function that was deleted or renamed
  in `src/` (the old version keeps working in your notebook), and a
  module-level variable such as a run table (you keep the old data).
  `rf.<name>` avoids both. `%autoreload 3` also updates imported variables,
  and it adds new functions to your notebook.
- The editable install only knows the module names listed in `py-modules`.
  A new `src/*.py` file gives `ModuleNotFoundError` until you add it there
  and reinstall.
- A uv-made `.venv` has no `pip`, so `%pip install` in a notebook fails. Use
  `uv add` in a terminal instead.
- If something still behaves oddly after a reload, restart the kernel.

## Using the code

Each file in `src/` is a module you import by name:

| Module | What it is for |
|---|---|
| `daq` | n26 and later: load acquisitions by number, scans, and the run log |
| `spectrum` | `Spectrum`, an immutable (energy, intensity, meta) value, and `combine` / `average` |
| `droplet_temperature` | `DropletCooling`: Knudsen-model temperature of an evaporatively cooled microdroplet vs. travel distance |
| `calibration` | Energy scales: fit an elastic sweep, save a scale, look one up |
| `fitting_functions` (`ff`) | lmfit peak models and `residual` |
| `RIXS_functions` (`rf`) | The older API for d22, m23, f24 and m24: `filter_samples`, `load_from_index`, `RIXS_run` |
| `beamtimes` | The registry of those older beamtimes |
| `veritas_paths` | Where files are: repo folders and the paths in `config.local.toml` |

### n26 data

Acquisitions are addressed by their number alone. The DAQ starts a new raw
file (`W47_<YYYYMMDD>_RIXS.h5`) now and then, but acq numbers keep counting
across files, and `daq` finds the file that holds each acquisition or scan.
`meta["day"]` says which file it came from.

```python
import daq
import spectrum as sp

daq.list_acqs()                                # one row per acq: energies, times, sample, comments
s = daq.load_acq(38, calibrated=True)          # light-window counts (not dark-corrected), in eV
s = s.smooth_gauss(0.05).normalised()          # every operation returns a new Spectrum
s.plot()

runs = [daq.load_acq(a, calibrated=True) for a in (38, 40)]
total = sp.combine(runs)                       # sums counts; normalise afterwards
xas = daq.load_scan(10519)                     # aemslit_ch1 against the scanned motor
```

- `calibrated=True` uses the energy scale `energy_scales/index.toml` assigns
  to that acquisition. Without it you get the DAQ's own axis, which is
  channels before acq44 and the DAQ's arm-525 eV calibration from acq44 on.
  `s.meta["energy_scale"]` says which one you have.
- A scale is only valid for the arm energy it was measured at. Loading an
  acquisition taken at another arm energy, or one where the arm moved during
  the acquisition, gives a warning.
- `dark_correct=True` returns the DAQ's own light − dark spectrum instead. It
  removes the dark-window background but adds that window's Poisson noise, so
  it is off by default; `s.meta["dark_corrected"]` says which one you have.
- An acquisition that is still running is rebuilt from the events recorded
  so far, with a warning.
- To merge acquisitions use `sp.combine`, which sums counts, so a 2 h run
  counts more than a 5 s one. `sp.average` is a plain mean.
- `daq.list_acqs` reads the run log CSV; refresh it with
  `python tools/fetch_runlog.py n26`.

### Fitting

```python
import lmfit
w = s.cropped(522, 527)                               # fit window, in eV
params = ff.make_parameters([524.0, 525.5],           # initial positions
                            amp=(0.1, 0, 10),         # (value, min, max), in units of the data
                            width=(0.5, 0.01, 3),     # Gaussian FWHM, eV
                            position_window=1.0)      # each position may move +-1 eV
result = lmfit.minimize(ff.residual, params, args=(w.energy, w.intensity, 2, "gauss"))
lmfit.report_fit(result)
```

`make_parameters`' default `amp` bounds suit one particular unnormalised
spectrum, so set `amp` for your data.

Line shapes are in `ff.PROFILES` (`gauss`, `voigt`, `pseudo_voigt`). The old
`ff.sum_gauss` / `sum_voigt` / `sum_pseudo_voigt` square the residual, so
lmfit minimised r⁴ and reported uncertainties that are far too large. They
still work but warn. Use `ff.residual`.

### New energy calibration

After an elastic sweep (beamline energy stepped during one acquisition):

```python
import calibration
scale, report = calibration.calibrate_elastic_sweep(43, order=2)
report            # one row per energy step: fitted channel, uncertainty, residual
calibration.save_energy_scale(scale, "n26", first_acq=43,
                              arm_energy=report.attrs["arm_energy"],
                              reference="elastic sweep acq43")
```

`save_energy_scale` writes a new `.npy` and adds its `index.toml` entry. It
never overwrites one. Open a pull request with both files.

### Before a pull request

```bash
uv run python -c "import RIXS_functions, fitting_functions"
uv run pytest
uv run ruff check src tests
```

## Working together

Branches, pull requests, and the rules for shared calibrations and run
tables are in [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).
