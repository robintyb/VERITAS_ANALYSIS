"""Beamtimes whose runs are rows of a run table (d22, m23, f24, m24).

Adding such a beamtime is one Beamtime entry below plus its energy scales in
energy_scales/index.toml. Beamtimes read by (day, acq) live in `daq` instead.
"""
from __future__ import annotations

import bisect
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from veritas_paths import ENERGY_SCALES, RUNINFO, raw_data_dir


@dataclass(frozen=True)
class Beamtime:
    key: str
    run_table: str                         # file in runinfo_files/
    dld_files: tuple[tuple[int, str], ...]  # (first run-table row, raw .h5 file name)
    temperature_column: str = "Temperature (K)"


BEAMTIMES: dict[str, Beamtime] = {b.key: b for b in [
    Beamtime("d22", "runinfo-Table 1.csv",
             dld_files=((0, "20221207_DLD_1.h5"), (68, "20221209_DLD_1.h5"),
                        (94, "20221210_DLD_1.h5"), (120, "20221211_DLD_1.h5"))),
    Beamtime("m23", "Veritas jet march 2023 - runinfo-2.xlsx",
             dld_files=((0, "20230329_DLD_1.h5"), (27, "20230330_DLD_1.h5"),
                        (37, "20230401_DLD_1.h5"), (55, "20230401_DLD_2.h5"),
                        (95, "20230404_DLD_1.h5"), (105, "20230405_DLD_1.h5"),
                        (114, "20230405_DLD_2.h5"), (134, "20230407_DLD_1.h5"))),
    Beamtime("f24", "VERITAS FEB 2024 - runinfo.xlsx",
             dld_files=((0, "20240220_DLD1.h5"), (56, "testdld.h5"))),
    # row 26 is the "filepath changed to 20240309_dld1.h5" note, so acq120 (row 25) is in the first file
    Beamtime("m24", "Veritas water jet vol 2 03-2024 .xlsx",
             dld_files=((0, "20240307_DLD1.h5"), (26, "20240309_DLD1.h5")),
             temperature_column="Temperature (C, Thermo Connection)"),
]}


def get(key: str) -> Beamtime:
    try:
        return BEAMTIMES[key]
    except KeyError:
        raise ValueError(f"unknown beamtime {key!r}; known: {tuple(BEAMTIMES)}. "
                         "For n26 and later use daq.load_acq(day, acq).") from None


@cache
def _read_table(path: Path, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)


def run_table(key: str) -> pd.DataFrame:
    """The beamtime's run table, re-read whenever the file changes."""
    path = RUNINFO / get(key).run_table
    return _read_table(path, path.stat().st_mtime).copy()


def dld_file(key: str, row: int) -> Path:
    segments = get(key).dld_files
    first_rows = [first for first, _ in segments]
    if row < first_rows[0]:
        raise IndexError(f"row {row} is before the first {key} file")
    return raw_data_dir(key) / segments[bisect.bisect_right(first_rows, row) - 1][1]


def read_index() -> dict[str, list[dict]]:
    """energy_scales/index.toml: beamtime key -> list of entries, in file order."""
    with open(ENERGY_SCALES / "index.toml", "rb") as f:
        return tomllib.load(f)


@cache
def _load_npy(name: str) -> np.ndarray:
    a = np.load(ENERGY_SCALES / name)
    a.flags.writeable = False
    return a


def energy_scale_entry(key: str, row: int) -> dict:
    get(key)
    for entry in read_index().get(key, []):
        if entry["first_row"] <= row <= entry.get("last_row", row):
            return entry
    raise LookupError(f"no entry in energy_scales/index.toml covers {key} row {row}")


def energy_scale(key: str, row: int) -> np.ndarray:
    """The calibrated energy scale (eV) of a run-table row, offset included."""
    entry = energy_scale_entry(key, row)
    return _load_npy(entry["file"]) + entry.get("offset_eV", 0.0)
