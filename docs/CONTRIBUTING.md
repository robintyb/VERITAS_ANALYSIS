# Working together during the beamtime

**Everyone works on their own branch. `main` holds only working code, and
everyone uses it. Nobody commits to `main` directly.**

First-time setup is in the [README](../README.md).

## 1. Your branch

Create it once:

```bash
git switch main && git pull
git switch -c <your-name>
git push -u origin <your-name>
```

Commit to it often, with messages that say what changed.

## 2. Get the latest `main`

At the start of every shift, and before opening a pull request:

```bash
git switch <your-name>
git fetch && git merge origin/main
```

With `%autoreload 2` the new code loads without restarting the kernel.
If something behaves oddly, restart the kernel.

## 3. Put your work into `main`

1. Get the latest `main` (section 2) and check that
   `python -c "import RIXS_functions, fitting_functions"` still works
   and `uv run pytest` passes.
2. `git push`, then open a pull request into `main` on GitHub.
3. Merge it with **"Create a merge commit"** (not squash or rebase).
   Ask someone to look at it if anyone is awake.

## 4. Rules

- **`src/`**: shared code. Don't change what an existing function takes or
  returns; add an optional argument or a new function. No personal paths;
  those go in `config.local.toml`.
- **`analysis_notebooks/<your-name>/`**: edit only your own notebooks.
  Copy someone else's into your folder to build on it.
- **`runinfo_files/`**: edit the `.csv`, not the `.xlsx`. If a merge
  conflicts because two people added rows, keep both.
- **`energy_scales/`**: an energy scale is valid only for the arm energy
  it was measured at, so a new elastic sweep usually means a new scale.
  Make it with `calibration.calibrate_elastic_sweep(acq)`, check the
  fit report, and save it with `calibration.save_energy_scale(...)`, which
  adds a new file and its `energy_scales/index.toml` entry. Never overwrite
  an existing `.npy`: older ones must stay unchanged so old results can be
  reproduced. Tell everyone when you add one.
- **Where code goes in `src/`**: a new processing step is a `Spectrum`
  method or a function that takes a `Spectrum`; a new line shape is a
  `PROFILES` entry in `fitting_functions.py`; a new beamtime is data
  (`config.local.toml`, `beamtimes.py`, `index.toml`), not new `if`
  branches. Anything about one sample or edge belongs in your notebook.

Changes to run tables and calibrations go through a pull request like
code. Keep those small and merge them quickly, since others are waiting.

## 5. Using JupyterHub

The Git panel (branch icon in the left sidebar) shows your current branch
and lets you switch branches, stage, commit, push, and pull.

- Check the panel shows **your** branch, not `main`, before committing.
- For merging `main` into your branch (section 2), open a terminal
  (**File → New → Terminal**) and run the commands there.
- Open pull requests on the GitHub web page.

## 6. Traceability

When you save a figure or result you may use later, include the output of
`git rev-parse --short HEAD` in the file name or plot title.

## 7. When something goes wrong

| Problem | Fix |
|---|---|
| Conflict in a `.py` file | Keep the right lines between `<<<<<<<` and `>>>>>>>`, then `git add <file>` and `git commit` |
| Conflict in a notebook or `.npy` | `git checkout --ours <file>` (keep yours) or `--theirs` (keep incoming), then `git add` and `git commit` |
| Merge got messy | `git merge --abort` |
| Committed to `main` by mistake, not pushed | `git switch -c <your-name>-rescue`, then `git switch main && git reset --hard origin/main` |
| `main` is broken | Tell everyone and fix it in a small pull request right away |
