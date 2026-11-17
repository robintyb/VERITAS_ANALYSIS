"""Check that load_from_index opens the right raw file for every run-table row.

Usage:
    python tools/check_run_mapping.py <beamtime>      # e.g. m23

For every row whose run name looks like ``acq`` plus three digits, this calls
``RIXS_functions.load_from_index`` (the same call notebooks make) and compares
the energies recorded in the opened file against the run table:

    ok                   beamline energy within 0.3 eV of the table, and arm
                         energy within 0.5 eV (or either arm value is nan)
    energy mismatch      beamline energy disagrees with the table
    arm mismatch         beamline energy agrees, arm energy disagrees
    no table energy      the table has no beamline energy, nothing compared
    missing acquisition  load_from_index raised KeyError (acquisition not in file)
    error                any other exception (type and message printed)

It uses only the public API, so it checks what callers see. Rerun it whenever
a beamtime's run table or file table changes. It needs the beamtime's raw-data
path in config.local.toml, and reading the raw files can take minutes.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import RIXS_functions as rf

ENERGY_TOL = 0.3  # eV
ARM_TOL = 0.5  # eV; files record 526.30 where the m23 table says 526
RUN_NAME = re.compile(r'acq\d{3}')
STATUSES = ('ok', 'energy mismatch', 'arm mismatch', 'no table energy',
            'missing acquisition', 'error')


def table_value(value):
    """A run-table energy as float; handles '535 eV' strings and blanks."""
    try:
        return float(str(value).replace('eV', '').strip())
    except ValueError:
        return np.nan


def check_row(beamtime, row, table_energy, table_arm):
    """Return (status, file name, detail) for one run-table row."""
    try:
        run = rf.load_from_index(row, runinfo=beamtime)
    except KeyError as e:
        return 'missing acquisition', '', f'KeyError: {e}'
    except Exception as e:
        return 'error', '', f'{type(e).__name__}: {e}'

    name = Path(run.filename).name
    energy, arm = run.beamline_energy, run.arm_energy
    detail = (f'table {table_energy:7.2f} / {table_arm:7.2f}   '
              f'file {energy:7.2f} / {arm:7.2f}')
    if np.isnan(table_energy):
        return 'no table energy', name, detail
    if not abs(energy - table_energy) <= ENERGY_TOL:  # also catches nan
        return 'energy mismatch', name, detail
    if np.isnan(arm) or np.isnan(table_arm) or abs(arm - table_arm) <= ARM_TOL:
        return 'ok', name, detail
    return 'arm mismatch', name, detail


def main(beamtime):
    if beamtime not in rf.BEAMTIMES:
        sys.exit(f'beamtime must be one of {rf.BEAMTIMES}')
    table = getattr(rf, f'runinfo_{beamtime}')

    counts = Counter()
    print(f'{"row":>4}  {"run":7}  {"file":20}  {"beamline / arm energy (eV)":49}  status')
    for row, entry in table.iterrows():
        run_name = str(entry['Run name']).strip()
        if not RUN_NAME.fullmatch(run_name):
            continue
        status, name, detail = check_row(
            beamtime, row,
            table_value(entry['Veritas beamline energy (eV)']),
            table_value(entry['X-ray energy (eV) Veritasrm_energy']))
        counts[status] += 1
        print(f'{row:4d}  {run_name:7}  {name:20}  {detail:49}  {status}', flush=True)

    print(f'\n{beamtime}: {sum(counts.values())} rows checked')
    for status in STATUSES:
        print(f'  {status:20} {counts[status]}')
    return 0 if counts['ok'] == sum(counts.values()) else 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
