# Working together during the beamtime

**Everyone works on their own branch. `main` holds only code that works,
and everybody uses it.**

First-time setup (install, `pre-commit install`, `config.local.toml`) is in
the [README](../README.md).

## 1. Your branch

Create it once, named after you:

```bash
git switch main && git pull
git switch -c <your-name>
git push -u origin <your-name>
```

Work and commit on it as often as you like. Commit small steps with clear
messages ("add background subtraction to RIXS_run", not "update").

## 2. Stay in sync with `main`

At the **start of every shift**, and **before every merge into `main`**,
bring `main` into your branch:

```bash
git switch <your-name>
git fetch
git merge origin/main
```

Do this from your own branch. Doing `git pull` on `main` alone does
**not** update your branch.

## 3. Get your work into `main`

When a function or feature works and others could use it:

1. Sync with `main` (step 2) and check that nothing broke:
   ```bash
   python -c "import RIXS_functions, fitting_functions"
   ```
   If you changed something people already use, also run your notebook
   from top to bottom.
2. Push your branch and open a pull request into `main`:
   ```bash
   git push
   gh pr create --base main --fill      # or use the GitHub web page
   ```
3. Merge it with **"Create a merge commit"**, not squash or rebase.
   Personal branches stay alive after merging, and squashing makes the same
   changes show up again in your next PR. If someone else is awake, ask them
   to glance at it. If not, merge it yourself.

Don't commit directly to `main`.

## 4. Get others' work

```bash
git switch <your-name>
git fetch && git merge origin/main
```

The package is installed in editable mode. With `%autoreload 2` in your
notebook, the new code is picked up without restarting the kernel. If
module-level variables changed, restart the kernel.

## 5. Rules per folder

**`src/`** (shared code)
- Don't change what an existing function takes or returns. Add an optional
  keyword argument with a default, or write a new function.
- Anything that more than one person needs belongs here, not in a notebook.
- Don't add personal paths. Raw-data locations go in `config.local.toml`.

**`analysis_notebooks/<your-name>/`**
- Edit only your own notebooks. To build on someone else's, copy it into
  your own folder.
- Outputs are stripped automatically on commit. Save figures to `outputs/`
  (which git ignores).

**Calibrations and run tables are shared.** Anyone can add or update them.
Every change goes into `main` through a PR, like code (section 3), so it
reaches everyone and git records who changed what and when. Keep these PRs
small and merge them quickly, because other people are waiting for new
runs and calibrations.

**`energy_scales/`**
- **Don't overwrite or rename an existing calibration**, even to improve
  it. Results already made with the old file must stay reproducible.
  Save it under the next free number, e.g. `escale_f24_3.npy`, and in the
  same PR:
  - add a row to `energy_scales/index.csv` saying which runs it applies to,
    any offset added on load, the reference used, who made it, the date it
    was added, and which notebook or function made it;
  - change the old row's `status` to `superseded by <new file>`;
  - update `src/` if the code should now use the new file.
- `git log -- energy_scales/` shows the full history.

**`runinfo_files/`**
- Edit the **`.csv`**, which is what the code reads. Treat `.xlsx` files as
  read-only, because git can't merge them.
- If two people add rows at the same time, the merge will conflict at the
  end of the file. Keep both sets of rows.
- Don't change the format (separator, column names, encoding) without
  checking that `import RIXS_functions` still works.
- `git log -p -- runinfo_files/` shows who changed which runs.

## 6. Make results traceable

At the end of each shift, the person on analysis duty tags `main`:

```bash
git switch main && git pull
git tag shift-2026MMDD-<am|pm|night> && git push --tags
```

When saving a figure or file you might use later, record the commit:
`git rev-parse --short HEAD` (e.g. put it in the file name or the plot title).

## 7. When something goes wrong

| Problem | Fix |
|---|---|
| Merge conflict in a `.py` file | Open the file, keep the right parts between `<<<<<<<` and `>>>>>>>`, then `git add <file>` and `git commit` |
| Conflict in a notebook or `.npy` | Choose one side: `git checkout --ours <file>` (yours) or `--theirs` (incoming), then `git add` and `git commit` |
| Merge got messy and you want out | `git merge --abort` |
| Committed to `main` by mistake (not yet pushed) | `git switch -c <your-name>-rescue`, then `git switch main && git reset --hard origin/main` |
| `main` is broken | Tell everyone, then fix it in a small PR straight away. Don't leave it for the next shift |

## Cheat sheet

```bash
git status                                  # where am I, what changed
git switch <your-name>                      # go to your branch
git add <files> && git commit -m "..."      # save a step
git fetch && git merge origin/main          # get everyone's latest
git push                                    # back up your branch on GitHub
gh pr create --base main --fill             # propose merging into main
```
