"""Download a beamtime's run log from Google Sheets and write it as a clean CSV.

Usage:
    python tools/fetch_runlog.py <beamtime>      # e.g. n26

The sheet id, tab and output path come from [runlog.<beamtime>] in
config.local.toml. The sheet must be shared as "anyone with the link can
view"; no Google login is used.

Output columns, in order:

    day          date of the raw file holding the run, as YYYYMMDD
    acq          N for a run named acqN (RIXS file)
    entry        N for a run named entryN (Scan file)
    uncertain    True when the run name ends in '?'
    ...          the sheet's columns, renamed through COLUMNS; unknown
                 headers pass through, slugified
    unparsed     numeric cells that are not a plain number, as name=text

`day` is found by looking the group up in the raw files. One file can run
past midnight, so the file date and the run's start date can differ. Rows are
taken to be in chronological order. When an acq number is lower than the one
before it, the run is looked up only in later files, so the mapping also
holds if the DAQ restarts its numbering in a new file. Runs not yet in any
raw file get an empty `day`; the next fetch fills it in.

The file is rewritten only when its content changes, through a temp file and
rename, so a reader never sees a half-written CSV and its mtime changes only
on real edits. If the download or a raw file fails, the old CSV is kept and
the script exits 1.
"""
import csv
import io
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime

import h5py

from veritas_paths import raw_data_dir, runlog_csv, runlog_sheet

COLUMNS = {
    'run name': ('run_name', str),
    'sample name': ('sample', str),
    'rixs or xas': ('technique', str),
    'veritas beamline energy': ('excitation_energy_ev', float),
    'step size': ('step_size', float),
    'x-ray energy': ('arm_energy_ev', float),
    'temperature': ('temperature_k', float),
    'x coordinates': ('x', float),
    'y coordinates': ('y', float),
    'z coordinates': ('z', float),
    'slit': ('slit_um', float),
    'time start': ('time_start', 'timestamp'),
    'comments': ('comments', str),
}
RUN_NAME = re.compile(r'(acq|entry)\s*(\d+)\s*(\?)?', re.IGNORECASE)
RAW_FILE = re.compile(r'_(\d{8})_(RIXS|Scan)\.h5$')
RAW_KIND = {'acq': 'RIXS', 'entry': 'Scan'}
TIMESTAMP_FORMATS = ('%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M')


def fetch(sheet_id, gid):
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.headers.get_content_type() != 'text/csv':
            raise ValueError('Google returned a non-CSV page; is the sheet shared as "anyone with the link"?')
        return response.read().decode('utf-8')


def column_for(header):
    key = header.strip().lower()
    for prefix, column in COLUMNS.items():
        if key.startswith(prefix):
            return column
    return re.sub(r'[^a-z0-9]+', '_', key).strip('_'), str


def parse_cell(text, kind):
    """The typed value, or None and a flag that the text did not parse."""
    if not text or kind is str:
        return text or None, True
    if kind is float:
        try:
            return float(text), True
        except ValueError:
            return None, False
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt).isoformat(), True
        except ValueError:
            pass
    return None, False


def parse_sheet(text):
    header, *body = csv.reader(io.StringIO(text))
    columns = [(i, *column_for(h)) for i, h in enumerate(header) if h.strip()]
    rows, skipped = [], 0
    for cells in body:
        cells = [c.strip() for c in cells]
        if not any(cells):
            skipped += 1
            continue
        row = {'day': None, 'acq': None, 'entry': None, 'uncertain': False}
        unparsed = []
        for i, name, kind in columns:
            value, ok = parse_cell(cells[i] if i < len(cells) else '', kind)
            row[name] = value
            if not ok:
                unparsed.append(f'{name}={cells[i]}')
        row['unparsed'] = '; '.join(unparsed) or None
        match = RUN_NAME.fullmatch(row.get('run_name') or '')
        if match:
            row[match[1].lower()] = int(match[2])
            row['uncertain'] = bool(match[3])
        else:
            print(f'warning: run name {row.get("run_name")!r} is not acqN or entryN', file=sys.stderr)
        rows.append(row)
    if skipped:
        print(f'skipped {skipped} blank rows', file=sys.stderr)
    return rows


def raw_index(raw_dir):
    """{'RIXS': [(day, group names), ...], 'Scan': [...]} in date order."""
    index = {'RIXS': [], 'Scan': []}
    for path in sorted(raw_dir.iterdir()):
        match = RAW_FILE.search(path.name)
        if match:
            with h5py.File(path, 'r', locking=False) as f:
                index[match[2]].append((int(match[1]), set(f.keys())))
    return index


def assign_days(rows, index):
    floor = {kind: 0 for kind in RAW_KIND}
    previous = {kind: -1 for kind in RAW_KIND}
    for row in rows:
        kind = 'acq' if row['acq'] is not None else 'entry' if row['entry'] is not None else None
        if kind is None:
            continue
        number, files = row[kind], index[RAW_KIND[kind]]
        start = floor[kind] + 1 if number < previous[kind] else floor[kind]
        previous[kind] = number
        found = next((i for i in range(start, len(files)) if f'{kind}{number}' in files[i][1]), None)
        if found is None:
            print(f'note: {row["run_name"]} is not in the raw files yet', file=sys.stderr)
            continue
        floor[kind] = found
        row['day'] = files[found][0]


def to_csv(rows):
    names = list(dict.fromkeys(name for row in rows for name in row))
    names.append(names.pop(names.index('unparsed')))
    out = io.StringIO()
    writer = csv.DictWriter(out, names, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def write_if_changed(path, text):
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)
    return True


def main(beamtime):
    rows = parse_sheet(fetch(*runlog_sheet(beamtime)))
    assign_days(rows, raw_index(raw_data_dir(beamtime)))
    path = runlog_csv(beamtime)
    changed = write_if_changed(path, to_csv(rows))
    print(f'{"wrote" if changed else "unchanged"} {path} ({len(rows)} rows)')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        main(sys.argv[1])
    except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
        sys.exit(f'error: {error}; kept the previous CSV')
