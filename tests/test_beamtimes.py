"""Registry and energy-scale index consistency. Needs no raw data."""
from veritas_paths import ENERGY_SCALES, RUNINFO

import numpy as np
import pytest

import beamtimes as bts

NOT_IN_REPO_YET = {"escale_ice_whole_detector.npy"}  # d22, see energy_scales/index.toml
INDEX = bts.read_index()


@pytest.mark.parametrize("key", bts.BEAMTIMES)
def test_dld_files_start_at_row_0_and_increase(key):
    firsts = [first for first, _ in bts.BEAMTIMES[key].dld_files]
    assert firsts[0] == 0
    assert firsts == sorted(set(firsts))


@pytest.mark.parametrize("key", bts.BEAMTIMES)
def test_run_table_has_the_columns_filter_samples_uses(key):
    assert (RUNINFO / bts.BEAMTIMES[key].run_table).exists()
    columns = bts.run_table(key).columns
    assert {"Run name", "Veritas beamline energy (eV)"} <= set(columns)


@pytest.mark.parametrize("key", bts.BEAMTIMES)
def test_every_run_table_row_has_exactly_one_energy_scale(key):
    entries = INDEX[key]
    for row in bts.run_table(key).index:
        covering = [e for e in entries if e["first_row"] <= row <= e.get("last_row", row)]
        assert len(covering) == 1, f"{key} row {row}: {[e['file'] for e in covering]}"


def test_every_index_file_exists():
    files = {e["file"] for entries in INDEX.values() for e in entries}
    missing = {f for f in files if not (ENERGY_SCALES / f).exists()}
    assert missing == NOT_IN_REPO_YET


def test_acq_entries_are_complete():
    for key, entries in INDEX.items():
        for e in entries:
            if "first_row" in e:
                continue
            assert {"file", "first_acq", "arm_energy"} <= set(e), (key, e)
            assert e.get("last_acq", e["first_acq"]) >= e["first_acq"], (key, e)


def test_f24_row_56_uses_the_second_scale():
    expected = np.load(ENERGY_SCALES / "escale_f24_2.npy") + 1.0
    assert np.array_equal(bts.energy_scale("f24", 56), expected)


def test_unknown_beamtime_raises():
    with pytest.raises(ValueError, match="known"):
        bts.get("feb24")
