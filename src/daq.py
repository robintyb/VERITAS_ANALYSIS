"""Reading the per-session DAQ files of current beamtimes (n26 onwards).

Raw files are `<prefix>_<day>_RIXS.h5` (groups `acq0..acqN`) and
`<prefix>_<day>_Scan.h5` (groups `entryNNNNN`) in `raw_data_dir(beamtime)`.
`day` is the YYYYMMDD in the file name. A file is one DAQ session and can
run past midnight, so an acquisition is identified by (day, acq).

The DAQ appends to the RIXS file while people analyse it, so every file is
opened read-only without HDF5 file locking.
"""
from __future__ import annotations

import copy
import warnings
from datetime import datetime
from functools import cache
from pathlib import Path

from veritas_paths import raw_data_dir, runlog_csv  # before h5py, see veritas_paths

import h5py
import numpy as np
import pandas as pd
from rixsaqlib.dldcalibration import DLDCalibration

from spectrum import Spectrum


def open_h5(path) -> h5py.File:
    return h5py.File(path, "r", locking=False)


def _daq_file(beamtime: str, day, kind: str) -> Path:
    folder = raw_data_dir(beamtime)
    matches = sorted(folder.glob(f"*_{day}_{kind}.h5"))
    if len(matches) != 1:
        found = ", ".join(m.name for m in matches) or "none"
        raise FileNotFoundError(f"expected one *_{day}_{kind}.h5 in {folder}, found {found}")
    return matches[0]


def rixs_file(day, beamtime: str = "n26") -> Path:
    return _daq_file(beamtime, day, "RIXS")


def scan_file(day, beamtime: str = "n26") -> Path:
    return _daq_file(beamtime, day, "Scan")


def days(beamtime: str = "n26") -> list[str]:
    """The `day` of every RIXS file in the beamtime's raw-data folder."""
    return sorted(p.name.split("_")[-2] for p in raw_data_dir(beamtime).glob("*_RIXS.h5"))


def _position(group, name):
    try:
        return float(group[f"{name}/position"][()])
    except KeyError:
        return np.nan


def _time(group, name):
    if name not in group:
        return None
    return datetime.fromisoformat(group[name][()].decode())


MANIPULATOR = "a_mp1_"
TIMEZONE = "Europe/Stockholm"  # the DAQ's start_time/end_time are local; its logs are unix time


def _log(group, name) -> np.ndarray:
    """(unix time, value) rows of External/<name>; empty if there is no log or it can't
    be read yet (a running acquisition: the DAQ has written past our view of the file)."""
    try:
        return group[f"External/{name}"][()]
    except (KeyError, OSError):
        return np.empty((0, 2))


def _manipulator(group) -> dict:
    """{a_mp1_<axis>: position} for every sample-manipulator axis: the mean of the
    acquisition's External log (sampled about once a second while it counts), or the
    startmetadata value where there is no readable log (a_mp1_tilt, running acquisitions)."""
    start = group["startmetadata"]
    positions = {}
    for axis in sorted(name for name in start if name.startswith(MANIPULATOR)):
        log = _log(group, f"{axis}/position")
        positions[axis] = float(log[:, 1].mean()) if len(log) else _position(start, axis)
    return positions


def manipulator_log(day, acqs=None, beamtime: str = "n26") -> pd.DataFrame:
    """The sample-manipulator positions logged during the acquisitions, one row per sample.

    Columns: time (datetime64, local like start_time), acq, and one column per a_mp1_<axis> that has an
    External log (azim, x, y, yaw, z), in the DAQ's units. The log is sampled about once a
    second while an acquisition counts, so there are gaps between acquisitions.
    `acqs` selects acquisitions (default all); one whose log is not readable is skipped.
    """
    frames = []
    with open_h5(rixs_file(day, beamtime)) as f:
        for acq in sorted(int(n[3:]) for n in f if n.startswith("acq")) if acqs is None else acqs:
            group = f[f"acq{acq}"]
            axes = {}
            for axis in sorted(n for n in group["startmetadata"] if n.startswith(MANIPULATOR)):
                log = _log(group, f"{axis}/position")
                if len(log):
                    axes[axis] = pd.Series(log[:, 1], index=pd.to_datetime(log[:, 0], unit="s", utc=True).tz_convert(TIMEZONE).tz_localize(None))
            if axes:
                frames.append(pd.DataFrame(axes).rename_axis("time").reset_index().assign(acq=acq))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def acq_meta(group) -> dict:
    """Start/end metadata. The start values are saved before the acquisition and can
    be stale (n26 acq35: the arm reached 528 eV just before counting, start says 523),
    so arm_energy_end is kept too; it is nan while an acquisition is running.
    The a_mp1_<axis> manipulator positions are averages over the acquisition."""
    start, end = group["startmetadata"], group.get("endmetadata", {})
    return {
        "beamline_energy": _position(start, "beamline_energy"),
        "arm_energy": _position(start, "veritasarm_energy"),
        "arm_energy_end": _position(end, "veritasarm_energy"),
        "start_time": _time(group, "start_time"),
        "end_time": _time(group, "end_time"),
        **_manipulator(group),
    }


def _scalars(group) -> dict:
    """{"motor/field": value} for every dataset under group; bytes decoded to str."""
    values = {}

    def add(name, obj):
        if isinstance(obj, h5py.Dataset):
            value = obj[()]
            values[name] = value.decode() if isinstance(value, bytes) else value
    group.visititems(add)
    return values


def acq_metadata(group) -> dict:
    """Everything the DAQ saved about an acquisition besides the detector data:
    "startmetadata" and "endmetadata" as {"motor/field": value} (endmetadata is empty
    while the acquisition runs), and "external" as {"motor/field": array} of
    (unix time, value) rows logged during the acquisition. The External log of a
    running acquisition is usually not readable yet; it is then {} and a warning says so."""
    def read(name):
        return _scalars(group[name]) if name in group else {}
    try:
        external = read("External")
    except (OSError, RuntimeError) as e:
        warnings.warn(f"{group.name}: External log not readable yet ({e}); left empty", stacklevel=3)
        external = {}
    return {"startmetadata": read("startmetadata"), "endmetadata": read("endmetadata"),
            "external": external}


def arm_moved(meta: dict) -> bool:
    return abs(meta["arm_energy_end"] - meta["arm_energy"]) > 0.01


def _calibration(group) -> DLDCalibration:
    """The acquisition's detector calibration (as rixsaqlib's loadCalibration), read without locking."""
    c = group["Instrument/DLD8080/calibration"]
    cal = DLDCalibration(c["x_size"][()], c["y_size"][()], c["t_size"][()])
    cal.energy_scale = c["energy_scale"][()]
    cal.set_curvature_center(c["curvature_center"][()])
    cal.set_curvatures_fromarray(c["curvatures"][()])
    cal.set_energylist_fromarray(c["energy_scale_lines"][()])
    cal.set_energy_scale_order(c["energy_scale_order"][()])
    cal.set_time_windows(c["time_window"][()])
    cal.masks = c["clipping_masks"][()]
    cal.set_time_modulo(c["time_modulo"][()])
    cal.set_time_modulo_offset(c["time_modulo_offset"][()])
    return cal


def dark_corrected_histogram(group):
    """A function points -> light - dark channel histogram, computed as the DAQ computes calib_x_spectrum."""
    light = _calibration(group)
    dark = copy.deepcopy(light)
    dark.set_time_windows(group["Instrument/DLD8080/calibration/dark_time_window"][()])
    n = light.x_size

    def histogram(points):
        def counts(cal):
            return np.bincount(cal.corrected_points(points)[:, 0].astype(int), minlength=n)[:n]
        return (counts(light) - counts(dark)).astype(float)
    return histogram


def acq_info(day, acq: int, beamtime: str = "n26") -> dict:
    """Beamline and arm energy, start and end time of one acquisition."""
    with open_h5(rixs_file(day, beamtime)) as f:
        return acq_meta(f[f"acq{acq}"])


def load_acq(day, acq: int, beamtime: str = "n26", calibrated: bool = False) -> Spectrum:
    """The DAQ's dark-corrected spectrum of one acquisition.

    On the axis the DAQ stored in data/energy_scale: detector channels, or eV once
    the DAQ has been given a calibration (energy_scale_lines; n26 from acq44 on).
    meta["energy_scale"] says which ("channels" or "daq"). With calibrated=True the
    axis is replaced by the scale energy_scales/index.toml assigns to this
    acquisition (meta["energy_scale"] names that file); with no such entry, the
    DAQ's eV axis is kept, and a channel axis raises LookupError. The DAQ writes the
    spectrum only when an acquisition ends, so for one still running it is
    rebuilt from the points recorded so far (meta["running"] is True).

    Besides the acq_meta summary (including the mean a_mp1_<axis> manipulator
    positions), meta holds all saved metadata: see acq_metadata.
    """
    path = rixs_file(day, beamtime)
    with open_h5(path) as f:
        group = f[f"acq{acq}"]
        energy = group["data/energy_scale"][()]
        meta = {"beamtime": beamtime, "day": str(day), "acq": int(acq), "file": str(path),
                **acq_meta(group),
                "energy_scale": "channels" if np.array_equal(energy, np.arange(energy.size)) else "daq",
                **acq_metadata(group)}
        meta["running"] = meta["end_time"] is None
        if meta["running"]:
            points = group["Instrument/DLD8080/data/points"][()]
            intensity = dark_corrected_histogram(group)(points)
            warnings.warn(f"acq{acq} is still running; spectrum rebuilt from the {len(points)} "
                          "points recorded so far", stacklevel=2)
        else:
            intensity = group["data/calib_x_spectrum"][()]
    meta["counts"] = float(intensity.sum())
    if arm_moved(meta):
        warnings.warn(f"acq{acq}: arm energy {meta['arm_energy']} eV at start, {meta['arm_energy_end']} eV at end",
                      stacklevel=2)
    if calibrated:
        from calibration import energy_scale_for  # calibration imports daq
        try:
            energy, entry = energy_scale_for(beamtime, day, acq)
            meta["energy_scale"] = entry["file"]
        except LookupError:
            if meta["energy_scale"] != "daq":
                raise
    return Spectrum(energy, intensity, meta)


def _scanned_motor(title: str, measurement) -> str:
    words = title.split()
    if len(words) > 1 and words[0].endswith("scan") and words[1] in measurement:
        return words[1]
    return "mono_energy_calib"


def load_scan(day, entry: int, channel: str = "aemslit_ch1", x: str | None = None,
              beamtime: str = "n26") -> Spectrum:
    """One channel of a Scan-file entry against the scanned motor (or `x`)."""
    path = scan_file(day, beamtime)
    with open_h5(path) as f:
        group = f[f"entry{entry}"]
        title = group["title"][()].decode()
        measurement = group["measurement"]
        x = x or _scanned_motor(title, measurement)
        meta = {"beamtime": beamtime, "day": str(day), "entry": int(entry), "file": str(path),
                "title": title, "x": x, "channel": channel}
        return Spectrum(measurement[x][()], measurement[channel][()], meta)


@cache
def _read_run_log(path: Path, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"acq": "Int64", "entry": "Int64", "day": "string"})


def run_log(beamtime: str = "n26") -> pd.DataFrame:
    """The run log written by tools/fetch_runlog.py, re-read whenever the CSV changes."""
    path = runlog_csv(beamtime)
    return _read_run_log(path, path.stat().st_mtime).copy()


_LOG_COLUMNS = ["sample", "comments"]


def _log_for(day: str, beamtime: str) -> pd.DataFrame:
    try:
        log = run_log(beamtime)
    except (KeyError, FileNotFoundError) as e:
        warnings.warn(f"no run log for {beamtime} ({e}); sample and comments left empty", stacklevel=3)
        return pd.DataFrame(columns=["acq", *_LOG_COLUMNS]).astype({"acq": "Int64"})
    log = log[(log["day"] == day) & log["acq"].notna()]
    return log.drop_duplicates("acq")[["acq", *_LOG_COLUMNS]]


def list_acqs(day, beamtime: str = "n26") -> pd.DataFrame:
    """One row per acquisition in a RIXS file, with sample and comments from the run log."""
    day = str(day)
    rows = []
    with open_h5(rixs_file(day, beamtime)) as f:
        for name in f:
            if not name.startswith("acq"):
                continue
            group = f[name]
            rows.append({"day": day, "acq": int(name[3:]), **acq_meta(group),
                         "n_points": group["Instrument/DLD8080/data/points"].shape[0]})
    acqs = pd.DataFrame(rows).sort_values("acq", ignore_index=True).astype({"acq": "Int64"})
    return acqs.merge(_log_for(day, beamtime), on="acq", how="left")
