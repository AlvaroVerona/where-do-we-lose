# Decision: two separate cost models, never mixed

`src/optimization/resource_allocation.py` keeps two models deliberately
apart:

1. **Analytical queueing proxy** (`expected_wait_hours`) — a deterministic
   version of `generate_data.py`'s own `compute_wait_hours` formula (same
   business-hours queueing curve, expected value of the noise term instead
   of a sampled draw). Cheap enough to evaluate thousands of times, which
   is what OR-Tools' CP-SAT needs while searching the staffing space
   (wired in via a precomputed lookup table + `add_element`, since the
   queueing curve is nonlinear in headcount).
2. **The real discrete-event simulation** (`generate_data.generate_dataset`),
   re-run per scenario via the `temporary_staffing` context manager, which
   mutates `STAGE_CONFIG`'s headcount for the duration of the `with` block
   and restores it afterward — even on exception (regression-tested:
   `test_temporary_staffing_restores_original_values_on_exception`).

**Every reported scenario metric** (SLA breach rate, P90, throughput,
utilization) comes from step 2. Step 1 only ever influences which staffing
plan the optimizer proposes — per the project's "do not invent these exact
values" rule, the proxy itself never appears in a result that gets
reported or compared.

## The budget constraint is what makes this a real optimization problem

Total headcount is capped at +5% over baseline (`budget_growth_fraction`).
Without a cap, the trivial answer is "hire unlimited staff at the
bottleneck" — not interesting, and not what the spec's own illustrative
example shows (some stages up, some down). With the cap, the solver is
forced into genuine tradeoffs, and the result reflects it: staff moves away
from over-staffed, low-utilization stages (Approval, Contract Generation,
Disbursement: ~−25%) toward the three genuinely constrained ones (Document
Validation, Credit Assessment, Risk Review: +22–30%) — not just the single
largest bottleneck. Confirmed the optimized plan beats the naive
"add-to-bottleneck-only" scenario on every reported metric while using
*fewer* additional employees (794 vs. 800) — see `README.md`'s Results
section.

## Bound tuning

First pass used `min_staff_fraction=0.5`/`max_staff_fraction=1.6` — produced
an extreme reallocation (Credit Assessment +60%, Approval −32%) that,
while mathematically valid under the objective, wasn't a palatable
business recommendation. Tightened to 0.75/1.3, which still produces a
genuine, non-trivial redistribution (±22–30%) without asking a department
to absorb a third of its headcount cut in one move.
