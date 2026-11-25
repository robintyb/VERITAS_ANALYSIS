import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fetch_runlog import parse_sheet

SHEET = """Run name,Temperature (K),Temperature setpoint (C),Comments
acq1,334.15,95,heated
acq2,295.15,,room
acq3,RT,,old row
"""


def test_temperature_k_comes_from_the_kelvin_column_and_the_setpoint_is_dropped():
    rows = parse_sheet(SHEET)
    assert [r["temperature_k"] for r in rows] == [334.15, 295.15, None]
    assert all("temperature_setpoint_c" not in r for r in rows)
    assert rows[2]["unparsed"] == "temperature_k=RT"
