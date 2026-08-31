# Decision: Resource utilization — peer-relative throughput, not hours-based

## First attempt (abandoned)

`hours worked / hours available`, where "hours worked" came from the
estimated processing-time split in `process_metrics.py`. That split is
measured in **elapsed calendar hours** between two consecutive events,
which are inflated relative to true business-hours effort by weekends and
off-hours (~3.4x, see `data/README.md`'s calendar-amplification note).
Dividing a calendar-hour quantity by a business-hours capacity is a unit
mismatch — and it produced **utilization above 100%** for Credit Assessment,
the busiest stage. Caught by inspecting the output table before trusting
it, not by a test.

## Final method

`calculate_resource_utilization` (`src/process_mining/bottleneck.py`)
avoids the mismatch entirely by working in **volume**, not hours: each
stage's average daily throughput per employee, expressed as a share of
that *same department's* P95 (peak) daily throughput per employee. Bounded,
self-calibrating per department, no hours-per-case assumption needed.

## The honest tradeoff

This measures "how close to this department's own best day does it run,
on average" — not absolute staffing tightness relative to other
departments. Every stage came back in a fairly narrow 76–82% band, even
though Credit Assessment is genuinely the most understaffed in the
underlying simulation (see `data/README.md`'s `STAGE_CONFIG` calibration
notes). This is because all departments see roughly the same *shape* of
weekly demand variability; the metric captures that shape, not the
absolute level. Documented directly in the function's docstring and in
`notebooks/03_bottleneck_analysis.ipynb`'s interpretation rather than
overclaiming what it shows.

See also: [[Backlog/Open questions]] — this is flagged there as something
a better cross-department benchmark could improve.
