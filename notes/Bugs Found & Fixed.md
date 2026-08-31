# Bugs found & fixed

Eight real issues, across every phase, all caught by actually running
things and inspecting output — not assumed correct because the code
looked right. Ordered by when they were found.

## 1. Wait-time calibration collapse at 50,000-instance scale (Phase 2)

A small-scale test (5k processes / 180 days) looked great — 17.5% SLA
breach rate. Scaling to the real 50k/546-day run gave **90% SLA breach**
and ~430h average processing time. Root cause: the small test's low daily
volume (~28/day blended) had accidentally kept utilization low; at real
scale (~92/day, ~110–140/day on business days), the queueing formula's
`u/(1-u)` term blew up. A second, compounding miscalibration: only ~10 of
24 hours/day and 5 of 7 days/week count as business hours, so 1 hour of
"effort" costs ~3.4 elapsed calendar hours on average — underestimated at
first. Fixed by redesigning staffing levels and SLA targets against that
amplification factor, iterating until the breach rate landed in a
realistic ~15–25% range. Documented in full in `data/README.md`.

## 2. Event-ordering validation that could never fail (Phase 3)

`check_event_ordering` in `src/data/validation.py` sorted the event log by
timestamp *before* checking whether it was sorted — trivially always true.
Caught by deliberately injecting a swapped-timestamp corruption and
noticing the check didn't flag it. Fixed by sorting on `event_id` (the
as-recorded order) instead, and checking timestamp monotonicity in that
order.

## 3. Rework rate misattributed to the wrong stage (Phase 4)

`compute_rework_events` derived a stage name by stripping " Rework" from
the activity name — `"Document Rework"` → `"Document"`, which doesn't match
the actual stage name `"Document Validation"`. Rework rate came back as 0%
for the three multi-word stages (Document Validation, Credit Assessment,
Risk Review) and nonzero only for the three single-word ones (Approval,
Contract Generation, Disbursement) — backwards from reality. Fixed by
using the real `REWORK_ACTIVITY` mapping instead of string manipulation.
Regression test: `test_rework_rate_attributed_to_correct_stage`.

## 4. Resource utilization exceeding 100% (Phase 5)

See [[Decisions/Resource Utilization Method]] for the full story — an
hours/calendar-time unit mismatch produced 216% utilization for Credit
Assessment. Caught by reading the output table, not a test. Replaced with
a peer-relative-throughput method that's bounded by construction.

## 5. `application_date` overflowing past `--end-date` (Phase 6)

Digital-channel applications assigned to the last day of the requested
range, with an hour near 23:59 plus up to 59 minutes, could roll into the
next calendar day — timestamps like `2025-07-01 00:46` showed up with
`--end-date 2025-06-30`. Fixed with an explicit clip to `end_date`'s last
second in `sample_arrival_timestamps`. Required regenerating the full
dataset and re-running every downstream notebook to stay consistent.

## 6. Streamlit rendering the dashboard in dark mode (Phase 10)

`st.plotly_chart`'s default `theme="streamlit"` ignored the custom Plotly
template entirely and inherited the OS/browser's dark mode, giving every
chart a black plot area against an otherwise light page. Fixed with an
explicit `.streamlit/config.toml` light theme plus `theme=None` on every
`st.plotly_chart` call, forcing the figure's own template to actually be
used.

## 7. Chart y-axis labels clipped to just "%" (Phase 10)

The shared Plotly template's margin (`l=10`) was too narrow for any
non-trivial tick label — percentages rendered as a bare "%" with the digits
clipped outside the visible plot area. Only caught by looking at an actual
screenshot, not the chart's data. Fixed with generous fixed margins
(`l=70, r=30, t=50, b=50`).

## 8. `'app' is not a package` / deprecated `use_container_width` (Phase 10–11)

Two separate Streamlit-specific issues: (a) `streamlit run app/app.py`
made `from app.components... import ...` fail with `'app' is not a
package` — fixed by adding `app/` itself to `sys.path` and importing
`components.*` directly, dropping the `app.` prefix everywhere. (b)
`use_container_width=True` is deprecated in the installed Streamlit
version (its own stated removal date had already passed) — replaced with
`width="stretch"` across all 19 call sites before it broke silently.
Caught the second one only because `tests/test_app.py` was written at all
— nothing else in the test suite ever imported `app/`.
