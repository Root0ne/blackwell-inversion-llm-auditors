# Reproducing the results

All commands run from the repository root and make no API calls. Runtimes were measured on
a Linux desktop (Python 3.14.4, numpy 2.3.5, scipy 1.16.3, matplotlib 3.10.7) under
`nice -n 15`; expect similar orders of magnitude elsewhere.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Data integrity

```bash
(cd data && sha256sum -c SHA256SUMS)                 # 16 files, instant
python -m unittest tests.test_data_integrity          # digests, counts, unique run keys, C3 merge; ~2 s
```

## Tests

```bash
python -m unittest discover -s tests -t .             # 92 tests, ~90 s (5 slow tests skipped)
RUN_SLOW=1 python -m unittest tests.test_reproducibility   # 5 tests, ~20 s
```

Most of the default run is the end-to-end dry run of 720 mock trials. The slow tests
regenerate every report below in a subprocess and compare it byte for byte, and render the
figures in both languages.

## Reports

| Report | Command | Runtime |
| --- | --- | ---: |
| Main report (Exp 1 and Exp 2, 7,631 trials) | `python -m blackwell_inversion.analysis.main_report --out-json X.json > X.txt` | 1.6 s |
| Mechanism analysis | `python -m blackwell_inversion.analysis.mechanism --out-json X.json > X.txt` | 6.5 s |
| Follow-up runs | `python -m blackwell_inversion.analysis.followup --out-json X.json > X.txt` | 2.5 s |
| Legacy stage report | `python -m blackwell_inversion.analysis.legacy_report data/main/results_c2_pilot.jsonl --json-out X.json --md-out X.md` | 0.8 s |

Compare with `cmp` against `reports/report_c3_unified.{json,txt}`,
`reports/report_mechanism.{json,txt}`, `reports/report_followup.{json,txt}` and
`reports/legacy/report_master_final.{json,md}`; all are byte-identical. Without
`--out-json`, `main_report` and `mechanism` overwrite the files in `reports/`.

`python -m blackwell_inversion.analysis.main_report --write-merged` rebuilds
`data/main/results_c3_expansion.jsonl` from `data/main/expansion_parts/`; the result is
byte-identical (checked by `tests/test_data_integrity.py`).

## Figures

```bash
python -m blackwell_inversion.analysis.figures --lang en --out-dir figures      # ~4 s
python -m blackwell_inversion.analysis.figures --lang tr --out-dir figures_tr   # Turkish manuscript
```

The figures are read from `reports/report_c3_unified.json` and `reports/report_followup.json`
and written as PDF and 600 DPI PNG. The `figures/` directory is not tracked.

## Dry run of the experiment pipeline

```bash
python -m blackwell_inversion --dry-run --experiment exp2 --models claude-haiku-5-5 \
    --exp2-seeds 1 --out /tmp/dry.jsonl                 # 8 trials, ~1 s
python -m blackwell_inversion --dry-run --experiment all --models claude-haiku-5-5 \
    --exp1-pairs 2 --exp2-seeds 2 --out /tmp/dry_all.jsonl   # 80 trials, ~25 s
```

The mock client calls the real oracle and runs real Python in the sandbox; it never sees
the ground truth. Records are identical across runs except `wall_s`.
