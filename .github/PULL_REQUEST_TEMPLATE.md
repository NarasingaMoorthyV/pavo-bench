## Summary

<!-- One-paragraph description of what this PR does. -->

## Type of change

- [ ] Bug fix
- [ ] Reproduction report (no code change)
- [ ] New experiment or extension
- [ ] Documentation / paper update
- [ ] Refactor (no behavior change)

## Checklist

- [ ] I've run `ruff check .` locally and it passes.
- [ ] If I changed a script whose output is committed (any `tier*.json`,
      `experiments/outputs/*.json`, `component_ablation_results.json`):
      - [ ] I re-ran the script and updated the committed JSON.
      - [ ] I included the delta-from-old-result in the PR description.
- [ ] If I added a new experiment: I added a row to the results table in
      `README.md` pointing to the new JSON and script.
- [ ] I've checked that `.github/workflows/validate.yml` passes on my branch.

## Associated issue

<!-- Closes #NNN, or N/A for small doc/typo fixes. -->
