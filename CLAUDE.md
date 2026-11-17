# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Analysis code for RIXS spectra from the VERITAS beamline (MAX IV), used interactively from Jupyter notebooks during beamtimes.

## Setup and checks

```bash
uv sync --extra dev        # not --group dev; plain `uv sync` drops the dev tools
uv run pre-commit install  # nbstripout strips notebook outputs on commit
cp config.example.toml config.local.toml   # per-user raw-data paths (git-ignored)
```

There are no tests or linter. The smoke check before merging is:

```bash
uv run python -c "import RIXS_functions, fitting_functions"
```

## Architecture

- `src/` is installed as flat top-level modules (not a package): `RIXS_functions`, `fitting_functions`, `veritas_paths`. Notebooks import them as `rf` / `ff` with `%autoreload 2`.
- `extern/rixsaqlib/` is a vendored copy of the VERITAS DLD (delay-line detector) reader, not on PyPI. Installed alongside `src/` via `pyproject.toml`.
- `veritas_paths.py` is the single source for file locations: repo files resolve from the repo root; raw `.h5` data dirs come from `[raw_data].<beamtime>` in `config.local.toml`. Raw data is never in the repo.
- Beamtimes are keyed by short codes: `d22`, `m23`, `f24`, `m24`, `n26`. `RIXS_functions` loads every run table and energy scale at import time (module-level globals), then:
  - `filter_samples(energy, runinfo=..., ...)` returns row indices from a beamtime's run table.
  - `load_from_index(index, runinfo=...)` maps a row index to the right `.h5` file (hardcoded file list and index cutoffs per beamtime) and returns a `RIXS_run`.
  - `RIXS_run` picks its energy scale by beamtime and row index in code. `energy_scales/index.toml` documents that mapping but is not read by the code, so keep the two in sync by hand.
- `fitting_functions.py` holds lmfit peak models (Gauss, Voigt, pseudo-Voigt sums). `pseudo_voigt` is defined three times; the last definition wins.

## Rules (from docs/CONTRIBUTING.md)

- Don't change the signature or return value of an existing `src/` function. Add an optional argument or a new function instead.
- No personal paths in `src/`; they go in `config.local.toml`.
- Never overwrite an `energy_scales/*.npy`. Add a new file plus an `index.toml` entry.
- Edit only notebooks in your own `analysis_notebooks/<name>/` folder; copy others' notebooks to build on them.
- Work on a personal branch and merge into `main` by pull request with a merge commit, not squash or rebase.
