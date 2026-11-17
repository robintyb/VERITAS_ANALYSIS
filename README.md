# Code for analysing RIXS spectra from VERITAS

## Layout

| Folder | Contents |
|---|---|
| `src/` | Shared analysis code (installed as a package, see below) |
| `analysis_notebooks/<your-name>/` | Your own notebooks. Don't edit someone else's notebook: copy it into your own folder. |
| `energy_scales/` | Energy calibrations (`.npy`). **Never overwrite one.** Add a new file and register it in `energy_scales/index.csv` |
| `runinfo_files/` | Run tables (`.xlsx` / `.csv`) |

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

`rixsaqlib` (the VERITAS DLD reader) is not on PyPI. Install it separately,
from the copy provided by the beamline.

In notebooks:

```python
%load_ext autoreload
%autoreload 2
import RIXS_functions as rf
import fitting_functions as ff
```

With an editable install and autoreload, a `git pull` takes effect without
restarting the kernel. If you change module-level variables, restart the
kernel anyway.

## Working together

A short collaboration guide will be added in `CONTRIBUTING.md`.
