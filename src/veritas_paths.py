"""Central place for file locations.

Files that live in the repository (energy scales, run tables) are located
relative to the repository root, so they work on every machine.
Raw data locations differ per user and are read from `config.local.toml`
at the repository root (see `config.example.toml`).
"""
import os
import tomllib
from pathlib import Path

# The DAQ appends to the current RIXS file while people analyse it, and
# rixsaqlib opens files itself. HDF5 reads this only when h5py is first
# imported, so modules that use h5py import veritas_paths before it.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

REPO_ROOT = Path(__file__).resolve().parents[1]
ENERGY_SCALES = REPO_ROOT / "energy_scales"
RUNINFO = REPO_ROOT / "runinfo_files"
LOCAL_CONFIG = REPO_ROOT / "config.local.toml"


def _load_config():
    if not LOCAL_CONFIG.exists():
        return {}
    with open(LOCAL_CONFIG, "rb") as f:
        return tomllib.load(f)


def raw_data_dir(beamtime):
    """Folder with the raw .h5 files for a beamtime key (e.g. 'm23')."""
    try:
        return Path(_load_config()["raw_data"][beamtime])
    except KeyError:
        raise KeyError(
            f"No raw-data path for '{beamtime}'. Copy config.example.toml to "
            f"{LOCAL_CONFIG.name} at the repo root and set [raw_data].{beamtime}."
        ) from None


def _runlog(beamtime):
    try:
        return _load_config()["runlog"][beamtime]
    except KeyError:
        raise KeyError(
            f"No run-log entry for '{beamtime}'. Copy the [runlog.{beamtime}] "
            f"block from config.example.toml into {LOCAL_CONFIG.name}."
        ) from None


def runlog_sheet(beamtime):
    """(sheet_id, gid) of the Google Sheet tab holding a beamtime's run log."""
    entry = _runlog(beamtime)
    return entry["sheet_id"], int(entry.get("gid", 0))


def runlog_csv(beamtime):
    """Clean run-log CSV written by tools/fetch_runlog.py. Relative paths are from the repo root."""
    return REPO_ROOT / _runlog(beamtime)["csv"]
