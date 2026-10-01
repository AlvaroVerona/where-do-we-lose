# Decision: Bottleneck score — weighted sum, not the spec's product formula

The spec's illustrative formula was:

```text
Bottleneck Score = normalized_waiting_time × normalized_volume × normalized_rework_rate
```

Went with a **weighted sum** instead, implemented in `calculate_bottleneck_score` (`src/process_mining/bottleneck.py`):

```text
score = 0.30·processing_time + 0.30·waiting_time + 0.20·volume
      + 0.10·rework_rate + 0.10·resource_utilization   (all min-max normalized)
```

## Why

A pure product is fragile: every stage's `rework_rate` here is a few
percent, so the least-rework stage normalizes to ~0 — and multiplying by
~0 crushes the whole score even for a stage that's genuinely slow and
overloaded. A weighted sum doesn't have that failure mode: a stage that's
slow but low-rework still scores appropriately high on its dominant
factors.

Demonstrated directly in `notebooks/03_bottleneck_analysis.ipynb` — computed
both formulas side by side on the real data and showed where the product
formula would have masked a real bottleneck that the weighted sum
correctly surfaces.

## Result

Confirmed stable across the whole build: Credit Assessment ranks #1 (score
0.756) in every re-run, re-verified after every subsequent data
regeneration (the `--end-date` overflow fix, see [[Bugs Found & Fixed]]).
`test_build_bottleneck_table_identifies_credit_assessment_as_top_bottleneck`
in `tests/test_process_metrics.py` is a standing regression test for this.
