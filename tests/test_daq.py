import h5py
import numpy as np
import pytest

import daq


def _acq_group(path, external=True):
    """An open acquisition group laid out like the n26 DAQ's, with a logged a_mp1_x."""
    with h5py.File(path, "w") as f:
        g = f.create_group("acq3")
        g["start_time"] = b"2026-11-19 08:01:29.545080"
        g["end_time"] = b"2026-11-19 08:09:04.869484"
        for name, x, arm in [("startmetadata", -0.98, 525.0), ("endmetadata", -1.05, 528.0)]:
            m = g.create_group(name)
            m["Sample"] = b"droplets"
            m["beamline_energy/position"] = 549.0
            m["veritasarm_energy/position"] = arm
            m["veritasarm_energy/focus"] = 0.0
            m["a_mp1_x/position"] = x
            m["a_mp1_tilt/position"] = 119.8
        if external:
            g["External/a_mp1_x/position"] = [[1.0, -1.0], [2.0, -1.1], [3.0, -1.2]]
            g["External/b316a-ea05/vac/vgcp-01/pressure"] = [[1.0, 1e-8]]
    return h5py.File(path, "r")["acq3"]


def test_acq_meta_averages_logged_manipulator_axes(tmp_path):
    meta = daq.acq_meta(_acq_group(tmp_path / "acq.h5"))
    assert meta["a_mp1_x"] == pytest.approx(-1.1)    # mean of the External log
    assert meta["a_mp1_tilt"] == 119.8               # not logged: start value
    assert (meta["arm_energy"], meta["arm_energy_end"]) == (525.0, 528.0)


def test_without_external_log_manipulator_falls_back_to_start_values(tmp_path):
    group = _acq_group(tmp_path / "acq.h5", external=False)
    assert daq.acq_meta(group)["a_mp1_x"] == -0.98
    assert daq.acq_metadata(group)["external"] == {}


def test_acq_metadata_keeps_everything_saved(tmp_path):
    meta = daq.acq_metadata(_acq_group(tmp_path / "acq.h5"))
    assert meta["startmetadata"]["Sample"] == "droplets"
    assert meta["startmetadata"]["veritasarm_energy/focus"] == 0.0
    assert meta["endmetadata"]["a_mp1_x/position"] == -1.05
    np.testing.assert_array_equal(meta["external"]["a_mp1_x/position"][:, 0], [1.0, 2.0, 3.0])
    assert meta["external"]["b316a-ea05/vac/vgcp-01/pressure"].shape == (1, 2)
