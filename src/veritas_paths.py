"""Central place for file locations.

Files that live in the repository (energy scales, run tables) are located
relative to the repository root, so they work on every machine.
Raw data locations differ per user and are read from `config.local.toml`
at the repository root (see `config.example.toml`).
"""
from pathlib import Path
import tomllib

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
