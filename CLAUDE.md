# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Analysis code for RIXS spectra from the VERITAS beamline (MAX IV), used interactively from Jupyter notebooks during beamtimes.

## Setup and checks

```bash
uv sync --extra dev        # not --group dev; plain `uv sync` drops the dev tools
uv run pre-commit install  # nbstripout strips notebook outputs on commit
cp config.example.toml config.local.toml   # per-user raw-data paths (git-ignored)
```

Checks before merging (ruff also runs as a pre-commit hook on `src/` and `tests/`):

```bash
uv run python -c "import RIXS_functions, fitting_functions"   # smoke
uv run pytest                                                  # no raw data needed
uv run ruff check src tests
```

## Architecture

- `src/` is installed as flat top-level modules (not a package), listed in `pyproject.toml` `[tool.setuptools] py-modules`. A new module must be added there and reinstalled with `uv sync --extra dev` before notebooks can import it; imports from inside `src/` work without that and hide the problem. Notebooks import `RIXS_functions as rf`, `fitting_functions as ff` with `%autoreload 2`.
- `extern/rixsaqlib/` is a vendored copy of the VERITAS DLD (delay-line detector) reader, not on PyPI.
- `veritas_paths.py` is the single source for file locations: repo files resolve from the repo root; raw `.h5` data dirs come from `[raw_data].<beamtime>` in `config.local.toml`, run-log CSVs from `[runlog.<beamtime>]`. Raw data is never in the repo. It also sets `HDF5_USE_FILE_LOCKING=FALSE`, which only takes effect before `h5py` is first imported; ruff's import order (a section of its own ahead of third-party) keeps `veritas_paths` first. Our own reads also pass `locking=False`, since the DAQ appends to the current file live.
- Two kinds of beamtime, keyed by short codes:
  - **Run-table beamtimes** `d22`, `m23`, `f24`, `m24`: `beamtimes.py` holds one `Beamtime` record each (run table, DLD file segments by row, temperature column). `RIXS_functions` keeps the old API on top of it: `filter_samples(energy, runinfo=...)` returns run-table rows, `load_from_index(row, runinfo=...)` returns a `RIXS_run`. Old module globals (`runinfo_m23`, `escale_f24_1`, ...) resolve lazily through `__getattr__`. Run tables are re-read when the file changes.
  - **Day/acq beamtimes** `n26` onwards: raw files are `<prefix>_<YYYYMMDD>_RIXS.h5` (groups `acqN`) and `_Scan.h5` (groups `entryN`). A file is one DAQ session, so an acquisition is `(day, acq)`. `daq.load_acq(day, acq)` returns the DAQ's own dark-corrected `data/calib_x_spectrum` on channels (`calibrated=True` for eV); `daq.list_acqs(day)` is the overview, joined with the run log (`tools/fetch_runlog.py`). `RIXS_run` raises for these keys.
- `energy_scales/index.toml` is the only map from runs to energy scales: rows for run-table beamtimes, `(day, first_acq..last_acq)` plus `arm_energy` for day/acq ones. `calibration.calibrate_elastic_sweep(day, acq)` fits a channel -> eV scale from an elastic sweep; `calibration.save_energy_scale` adds a new `.npy` and index entry.
- `spectrum.Spectrum` is an immutable `(energy, intensity, meta)` value; every operation returns a new one. `spectrum.combine` sums counts (use it to merge acquisitions), `spectrum.average` takes the mean.
- `fitting_functions.py`: line shapes in `PROFILES`; fit with `lmfit.minimize(ff.residual, params, args=(x, y, n, profile))`. The old `sum_*` functions return squared residuals (lmfit then minimises r**4) and warn.

## Where to add things

- New run-table beamtime: a `Beamtime` entry in `beamtimes.py` plus its `index.toml` entries.
- New day/acq beamtime: only `[raw_data]` (and `[runlog]`) in `config.local.toml`, then energy scales via `calibration`.
- New calibration: a new `.npy` and `index.toml` entry, never an edit to an existing `.npy`.
- New processing step: a `Spectrum` method or a function taking a `Spectrum`.
- New line shape: a `PROFILES` entry.
- Anything about one sample or edge: the notebook, not `src/`.

## Rules (from docs/CONTRIBUTING.md)

- Don't change the signature or return value of an existing `src/` function. Add an optional argument or a new function instead.
- No personal paths in `src/`; they go in `config.local.toml`.
- Never overwrite an `energy_scales/*.npy`. Add a new file plus an `index.toml` entry.
- Edit only notebooks in your own `analysis_notebooks/<name>/` folder; copy others' notebooks to build on them.
- Work on a personal branch and merge into `main` by pull request with a merge commit, not squash or rebase.
