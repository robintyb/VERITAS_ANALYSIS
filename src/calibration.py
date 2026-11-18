"""Energy calibration of the detector channels for beamtimes read through `daq`.

`calibrate_elastic_sweep` turns an acquisition during which the beamline
energy was stepped (an elastic sweep) into a channel -> eV scale.
`save_energy_scale` adds it to energy_scales/ with an index.toml entry, and
`energy_scale_for` finds the scale for an acquisition.

index.toml entries for these beamtimes are keyed by (day, acquisition range)
and record the arm energy the scale was measured at:

    [[n26]]
    file       = "escale_n26_1.npy"
    day        = "20261117"
    first_acq  = 37
    last_acq   = 45          # omit = to the end of that day's file
    arm_energy = 525.0
    offset_eV  = 0.0
    reference  = "elastic sweep acq37 (jet), 519-529 eV"

When several entries cover an acquisition, the one listed last wins.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass

from veritas_paths import ENERGY_SCALES

import lmfit
import numpy as np
import pandas as pd

import daq
import fitting_functions as ff
from beamtimes import read_index

ARM_ENERGY_TOLERANCE_EV = 0.01


@dataclass(frozen=True)
class Plateau:
    start: float   # unix time
    end: float
    energy: float  # median beamline energy on the plateau, eV


def plateaus(time, energy, tol: float = 0.05, min_duration: float = 30.0) -> list[Plateau]:
    """Stretches of a (time, energy) log where the energy holds still for min_duration seconds."""
    time, energy = np.asarray(time), np.asarray(energy)
    found: list[Plateau] = []
    for seg in np.split(np.arange(energy.size), np.flatnonzero(np.abs(np.diff(energy)) > tol) + 1):
        if time[seg[-1]] - time[seg[0]] < min_duration:
            continue
        e = float(np.median(energy[seg]))
        if found and abs(found[-1].energy - e) < tol:
            found[-1] = Plateau(found[-1].start, float(time[seg[-1]]), found[-1].energy)
        else:
            found.append(Plateau(float(time[seg[0]]), float(time[seg[-1]]), e))
    return found


def _fit_peak(channels, counts, window: int, profile: str):
    smoothed = np.convolve(counts, np.ones(25) / 25, mode="same")
    c0 = int(np.argmax(smoothed))
    sl = slice(max(c0 - window, 0), c0 + window)
    params = ff.make_parameters([c0], amp=(smoothed[c0], 0, np.inf), width=(40, 1, 10 * window),
                                gamma=(5, 1e-3, window) if ff.PROFILES[profile][1] else None,
                                position_window=window / 2)
    result = lmfit.minimize(ff.residual, params, args=(channels[sl], counts[sl], 1, profile))
    p = result.params
    return {"centre": p["position_0"].value, "centre_err": p["position_0"].stderr or np.nan,
            "fwhm": p["width_0"].value, "counts": counts[sl].sum()}


def calibrate_elastic_sweep(day, acq: int, beamtime: str = "n26", order: int = 1,
                            min_plateau_s: float = 30.0, profile: str = "gauss", window: int = 300):
    """Channel -> eV scale from an acquisition during which the beamline energy was stepped.

    Points are split by beamline-energy plateau (by timestamp), histogrammed with the
    file's own light and dark time windows, and one `profile` peak is fitted per
    plateau within +-`window` channels of its maximum. A polynomial of `order`
    maps peak centre -> beamline energy.

    Returns (scale, report): scale has one eV value per channel; report has one row
    per plateau, and report.attrs holds the polynomial and the acquisition's metadata.
    """
    path = daq.rixs_file(day, beamtime)
    with daq.open_h5(path) as f:
        group = f[f"acq{acq}"]
        points = group["Instrument/DLD8080/data/points"][()]
        log = group["External/beamline_energy/position"][()]
        channels = group["data/energy_scale"][()]
        meta = daq.acq_meta(group)
        histogram = daq.dark_corrected_histogram(group)

    rows = []
    for p in plateaus(log[:, 0], log[:, 1], min_duration=min_plateau_s):
        in_plateau = points[(points[:, 3] >= p.start) & (points[:, 3] <= p.end)]
        rows.append({"nominal_eV": p.energy, "start": p.start, "end": p.end, "n_points": len(in_plateau),
                     **_fit_peak(channels, histogram(in_plateau), window, profile)})
    if len(rows) <= order:
        raise ValueError(f"acq{acq}: {len(rows)} beamline-energy plateaus of at least {min_plateau_s} s "
                         f"found, need more than {order} for order {order}")
    report = pd.DataFrame(rows).sort_values("nominal_eV", ignore_index=True)

    coefficients = np.polyfit(report["centre"], report["nominal_eV"], order)
    report["fit_eV"] = np.polyval(coefficients, report["centre"])
    report["residual_eV"] = report["nominal_eV"] - report["fit_eV"]
    report.attrs = {"beamtime": beamtime, "day": str(day), "acq": int(acq), "order": order,
                    "coefficients": coefficients, "arm_energy": meta["arm_energy"]}
    return np.polyval(coefficients, channels), report


def _covers(entry: dict, day: str, acq: int) -> bool:
    return (entry.get("day") == day and entry["first_acq"] <= acq <= entry.get("last_acq", acq))


def energy_scale_for(beamtime: str, day, acq: int) -> tuple[np.ndarray, dict]:
    """(scale in eV, index entry) for one acquisition; warns if the arm energy differs."""
    day = str(day)
    matches = [e for e in read_index().get(beamtime, []) if _covers(e, day, acq)]
    if not matches:
        raise LookupError(f"no energy scale in energy_scales/index.toml covers {beamtime} day {day} acq{acq}")
    entry = matches[-1]
    arm = daq.acq_info(day, acq, beamtime)["arm_energy"]
    if not abs(arm - entry["arm_energy"]) <= ARM_ENERGY_TOLERANCE_EV:
        warnings.warn(f"acq{acq} was measured at arm energy {arm} eV, but {entry['file']} was calibrated "
                      f"at {entry['arm_energy']} eV", stacklevel=2)
    return np.load(ENERGY_SCALES / entry["file"]) + entry.get("offset_eV", 0.0), entry


def save_energy_scale(scale, beamtime: str, day, first_acq: int, arm_energy: float, reference: str,
                      last_acq: int | None = None, offset_eV: float = 0.0):
    """Write scale to a new energy_scales/escale_<beamtime>_<n>.npy and append its index.toml entry."""
    taken = [int(m.group(1)) for p in ENERGY_SCALES.glob(f"escale_{beamtime}_*.npy")
             if (m := re.fullmatch(rf"escale_{re.escape(beamtime)}_(\d+)\.npy", p.name))]
    path = ENERGY_SCALES / f"escale_{beamtime}_{max(taken, default=0) + 1}.npy"
    if not (np.isfinite(arm_energy) and np.isfinite(offset_eV)):
        raise ValueError(f"arm_energy ({arm_energy}) and offset_eV ({offset_eV}) must be finite")
    entry = {"file": path.name, "day": str(day), "first_acq": int(first_acq),
             "last_acq": None if last_acq is None else int(last_acq),
             "arm_energy": round(float(arm_energy), 3), "offset_eV": float(offset_eV), "reference": str(reference)}
    lines = [f"{k:<10} = {json.dumps(v)}" for k, v in entry.items() if v is not None]
    with open(path, "xb") as f:
        np.save(f, np.asarray(scale, dtype=float))
    with open(ENERGY_SCALES / "index.toml", "a") as f:
        f.write(f"\n[[{beamtime}]]\n" + "\n".join(lines) + "\n")
    return path
