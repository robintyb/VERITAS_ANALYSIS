import veritas_paths

import h5py
import pytest

import daq


@pytest.fixture
def config(tmp_path, monkeypatch):
    def write(text):
        path = tmp_path / "config.local.toml"
        path.write_text(text)
        monkeypatch.setattr(veritas_paths, "LOCAL_CONFIG", path)
    return write


def test_default_beamtime_comes_from_the_config(config):
    config('[default]\nbeamtime = "d27"\n')
    assert veritas_paths.default_beamtime() == "d27"
    assert veritas_paths.resolve_beamtime() == "d27"


def test_an_explicit_beamtime_beats_the_default(config):
    config('[default]\nbeamtime = "d27"\n')
    assert veritas_paths.resolve_beamtime("n26") == "n26"


def test_a_config_without_default_keeps_n26(config):
    config('[raw_data]\nn26 = "/x"\n')
    assert veritas_paths.default_beamtime() == "n26"


def test_daq_reads_the_default_beamtime_unless_told_otherwise(config, tmp_path, monkeypatch):
    config('[default]\nbeamtime = "d27"\n')
    folders = {}
    for key in ("d27", "n26"):
        folders[key] = tmp_path / key
        folders[key].mkdir()
        with h5py.File(folders[key] / "W1_20270101_RIXS.h5", "w") as f:
            f.create_group("acq0")
    monkeypatch.setattr(daq, "raw_data_dir", lambda beamtime: folders[beamtime])
    assert daq.acq_file(0).parent == folders["d27"]
    assert daq.acq_file(0, "n26").parent == folders["n26"]
    assert daq.days() == ["20270101"]
