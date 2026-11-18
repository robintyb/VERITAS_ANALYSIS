# Code for analysing RIXS spectra from VERITAS

## Layout

| Folder | Contents |
|---|---|
| `src/` | Shared analysis code (installed as a package, see below) |
| `analysis_notebooks/<your-name>/` | Your own notebooks. Don't edit someone else's notebook: copy it into your own folder. |
| `energy_scales/` | Energy calibrations (`.npy`). **Never overwrite one.** `energy_scales/index.toml` says which runs use which file |
| `runinfo_files/` | Run tables. The code reads the `.csv`; the `.xlsx` files are read-only |

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
removes the dev tools. Run `uv sync --extra dev` again after pulling
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

## Working together

Branches, pull requests, and the rules for shared calibrations and run
tables are in [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).
