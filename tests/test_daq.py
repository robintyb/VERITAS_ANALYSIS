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


@pytest.fixture
def raw_files(tmp_path, monkeypatch):
    """Two RIXS files whose acq numbers continue from one to the next, as n26's do."""
    for day, acqs in [("20261117", [0, 1]), ("20261120", [2])]:
        with h5py.File(tmp_path / f"W47_{day}_RIXS.h5", "w") as f:
            for acq in acqs:
                f.create_group(f"acq{acq}")
    monkeypatch.setattr(daq, "raw_data_dir", lambda beamtime: tmp_path)
    return tmp_path


def test_acq_file_finds_an_acquisition_in_whichever_file_holds_it(raw_files):
    assert daq.acq_file(1).name == "W47_20261117_RIXS.h5"
    assert daq.acq_file(2).name == "W47_20261120_RIXS.h5"
    assert daq._all_acqs("n26") == [0, 1, 2]


def test_acq_file_raises_for_a_missing_or_repeated_acquisition(raw_files):
    with pytest.raises(KeyError, match="acq7 .* found it in none"):
        daq.acq_file(7)
    with h5py.File(raw_files / "W47_20261121_RIXS.h5", "w") as f:
        f.create_group("acq2")
    with pytest.raises(KeyError, match="W47_20261120_RIXS.h5, W47_20261121_RIXS.h5"):
        daq.acq_file(2)


def test_acq_file_skips_a_file_the_daq_is_writing(raw_files):
    (raw_files / "W47_20261121_RIXS.h5").write_bytes(b"not hdf5 yet")
    assert daq.acq_file(1).name == "W47_20261117_RIXS.h5"
    with pytest.raises(KeyError, match="could not open W47_20261121_RIXS.h5"):
        daq.acq_file(7)


def test_manipulator_log_rejects_an_old_style_day(raw_files):
    with pytest.raises(TypeError, match="no longer takes a day"):
        daq.manipulator_log("20261117")


class _StubCalibration:
    """Keeps the points whose time (column 1) is inside the window; column 0 is the channel."""
    x_size = 4
    window = (0, 9)

    def set_time_windows(self, window):
        self.window = tuple(window)

    def corrected_points(self, points):
        return points[(points[:, 1] >= self.window[0]) & (points[:, 1] <= self.window[1])]


@pytest.fixture
def acq_with_points(tmp_path, monkeypatch):
    """acq3 with 5 light events (channels 0,1,1,2,3) and 2 dark events (channels 1,2)."""
    path = tmp_path / "W47_20261117_RIXS.h5"
    _acq_group(path).file.close()
    points = np.array([[0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [1, 20], [2, 21]], dtype=float)
    with h5py.File(path, "a") as f:
        g = f["acq3"]
        g["data/energy_scale"] = np.arange(4)
        g["data/calib_x_spectrum"] = [1.0, 1.0, 0.0, 1.0]
        g["Instrument/DLD8080/data/points"] = points
        g["Instrument/DLD8080/calibration/dark_time_window"] = [20, 29]
    monkeypatch.setattr(daq, "raw_data_dir", lambda beamtime: tmp_path)
    monkeypatch.setattr(daq, "_calibration", lambda group: _StubCalibration())


def test_load_acq_counts_the_light_window_unless_dark_correct_is_asked_for(acq_with_points):
    plain = daq.load_acq(3)
    np.testing.assert_array_equal(plain.intensity, [1, 2, 1, 1])
    assert plain.meta["counts"] == 5 and not plain.meta["dark_corrected"]

    corrected = daq.load_acq(3, dark_correct=True)
    np.testing.assert_array_equal(corrected.intensity, [1, 1, 0, 1])
    assert corrected.meta["counts"] == 3 and corrected.meta["dark_corrected"]
