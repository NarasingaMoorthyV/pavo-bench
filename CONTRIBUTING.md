# Contributing to PAVO-Bench

Thanks for thinking about contributing. The most useful contributions are:

1. **Reproduction reports** on model pairs we didn't test.
2. **Bug reports** when a script in `experiments/` doesn't match the numbers in the committed JSON results.
3. **Documentation fixes** — anything in the README or paper text that's unclear or out-of-date.
4. **New tier experiments** that extend the benchmark — e.g., additional noise conditions, non-English speech, alternative TTS back-ends.

## Reproducing results before filing an issue

Most "the numbers don't match" reports turn out to be model-version or hardware drift. Before filing an issue:

1. Run `bash experiments/setup.sh` to install pinned deps and pull the ollama model tags we used.
2. Run the specific experiment script that produced the JSON you're comparing against (see the README results table for the mapping).
3. Note your hardware: GPU model, driver, CUDA version, OS, Python version, and `pip freeze` output.

If your numbers are within one standard deviation of ours — reported inline in every `tier1_*.json` and `tier2_*.json` — consider that a successful reproduction.

## Filing a reproduction report

Use the "Reproduction report" issue template. Include:

- Hardware stack (GPU, driver, CUDA, host OS).
- Exact model tags you pulled (`ollama list` output for llama3.1 and gemma2).
- Which script you ran and the full command line.
- The resulting JSON (attach as a file, not inline).
- The tier number and metric you're comparing against.

## Filing a bug

Use the "Bug report" template. Smallest reproducer wins; a 5-line script that shows the bug is worth more than a 500-line wall of context.

## Pull requests

Open an issue first for anything that isn't a typo or doc fix. PRs are easier to land when the scope is agreed on in advance.

- Keep PRs focused. One logical change per PR.
- Add or update an experiment script alongside any code change that affects reported numbers.
- If your change regenerates a committed JSON, include the *new* JSON in the PR and note the delta from the old one in the description.
- The CI workflow (`.github/workflows/validate.yml`) checks that every committed JSON parses and that Python lints. PRs are easier to review when CI is green.

## Code style

- Python 3.10+.
- `ruff` for linting and formatting — the CI runs it.
- Type hints welcome but not required.
- Docstrings on any function that's part of an experiment entry point.

## Reporting security issues

Do not open a public issue for security reports. Email `moorthyv@sas.upenn.edu` directly.

## A note on attribution

This is an academic artifact. If you build on top of PAVO-Bench, please cite the TMLR paper (see `CITATION.cff`) rather than just linking to the repo.
