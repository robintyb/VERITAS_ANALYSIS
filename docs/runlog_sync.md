# Getting the n26 run log out of Google Sheets

During the November 2026 beamtime (`n26`) the run log is a Google Sheet that
people edit while measuring. `tools/fetch_runlog.py` turns its "Run table" tab
into a clean CSV that the analysis code can load, and maps every row to the
raw file it lives in.

## Running it

Add the run-log block from `config.example.toml` to your `config.local.toml`,
with the sheet id from the sheet's URL. The id stays out of git because the
repository is public and the id is enough to open the sheet.

```toml
[runlog.n26]
sheet_id = "<id>"
gid = 0
csv = "runinfo_files/n26_runlog.csv"
```

Then, from the repository root:

```bash
python tools/fetch_runlog.py n26
```

It prints `wrote <path> (N rows)` or `unchanged <path> (N rows)`. Code finds
the output through `veritas_paths.runlog_csv('n26')`; `daq.run_log('n26')`
loads it and `daq.list_acqs(day)` joins it to the acquisitions in a raw file. A notebook can refresh
before loading with `!python ../../tools/fetch_runlog.py n26`.

To keep the CSV current without thinking about it, run a loop in a terminal
or `tmux` for the length of the beamtime:

```bash
while true; do python tools/fetch_runlog.py n26; sleep 60; done
```

## What the sheet holds

The sheet has two tabs. "Run table" (`gid=0`) is the run log. "Shift Plan"
(`gid=561588040`) is the staffing grid and is ignored.

Rows are identified by `Run name` only. There is no date or file column.
`acqN` names a group in `W47_<YYYYMMDD>_RIXS.h5` and `entryN` names a scan in
`W47_<YYYYMMDD>_Scan.h5`. A file is one DAQ session, not one calendar day:
`W47_20261117_RIXS.h5` holds `acq0` to `acq45`, and `acq39` onwards started on
18 November. On 18 November only one RIXS file existed, so it is not yet known
whether acq numbering restarts with a new file.

The "X-ray energy (eV) Veritasrm_energy" column is the spectrometer arm
energy. For `acq38` the sheet says 525 and the file's `veritasarm_energy`
reads 524.99999994. "Veritas beamline energy" is the excitation energy, but
for energy maps it holds a range such as `519-527`, and once a list such as
`524,527, 529.5`. Temperature is mostly `RT`. Run names can carry a `?`
(`acq3?`). `time start` is filled in for one row.

## The CSV

| column | content |
|---|---|
| `day` | date of the raw file holding the run, `YYYYMMDD`; empty if the run is not in a file yet |
| `acq`, `entry` | the number from an `acqN` or `entryN` run name |
| `uncertain` | `True` when the run name ends in `?` |
| `run_name`, `sample`, `technique`, `comments` | text as typed |
| `excitation_energy_ev`, `arm_energy_ev`, `step_size`, `temperature_k`, `x`, `y`, `z`, `slit_um` | numbers; empty when the cell is not a plain number |
| `time_start` | ISO timestamp |
| `unparsed` | every numeric cell that did not parse, as `name=text; ...` |

A sheet column whose header is not in the script's `COLUMNS` table passes
through under a slugified name, so a column added mid-beamtime is not lost.
Blank rows are dropped. With pandas, read `acq` and `entry` as
`dtype={'acq': 'Int64', 'entry': 'Int64'}`, or they come back as floats.

`day` comes from listing the groups in each raw file, opened read-only with
`locking=False`. The script assumes the sheet is in chronological order. When
an acq number is lower than the previous row's, it looks only in later files.
That gives the right file whether or not numbering restarts. A run logged
before the DAQ has written it gets an empty `day`, and the next fetch fills it
in.

The CSV is rewritten only when its content changes, via a temp file and
`os.replace`. A reader never sees a half-written file, and the file's mtime
changes only on real edits, so a loader that re-reads on mtime does no extra
work. If the download fails or a raw file cannot be read, the script exits 1
and keeps the previous CSV.

## Options considered

Outbound HTTPS to Google works from the cluster node this was built on (the
export takes about 0.5 s). Compute nodes may differ.

**CSV export URL, polled (chosen).** One GET of
`/spreadsheets/d/<id>/export?format=csv&gid=<gid>` needs no credentials, only
"anyone with the link can view". It runs wherever Python and the raw data are.
Offline, it fails and the old CSV stays. If link sharing is turned off, Google
returns an HTML login page, which the script rejects (not tested; no private sheet was at hand).

The `gviz/tq?tqx=out:csv&sheet=<name>` route selects a tab by name, but it
guesses the header from column types and merged the two `entry` rows into the
header. Do not use it.

**Sheets API with a service account.** This needs a Google Cloud project, a
JSON key stored on the cluster, and the sheet shared with the service
account's address. It would keep working if link sharing were turned off, and
it can return typed values. For a sheet that is already shared by link, it
adds a credential to protect and gains nothing else. Offline, it fails the
same way.

**Push: Drive `files.watch` or Apps Script `onEdit`.** Both deliver an HTTP
request to an address you give them. Drive requires a public HTTPS URL with a
valid certificate. Apps Script could `UrlFetchApp` to any public URL. The
MAX IV offline cluster accepts no inbound connections from the internet
(inferred from its role; not tested), so neither can reach it without a relay
hosted elsewhere. A Drive notification also carries no content, so the
receiver would still fetch the CSV.

## A listener service: no

A push listener cannot receive anything on the cluster, and a relay outside
MAX IV is far more machinery than a 1 KB download. A polling service under
cron or systemd is also not worth it. The script is idempotent and cheap, so
running it on demand, or the shell loop above during shifts, gives the same
freshness with nothing to install, monitor, or clean up after the beamtime.
