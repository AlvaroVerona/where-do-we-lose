# Decision: predict SLA breach at submission time, not mid-process

The spec's "potential features" list (section 15) includes
`previous_stage_duration` and `number_of_reworks`, alongside intrinsic
application attributes. Deliberately **excluded both** from the final
feature set in `src/features/engineering.py`.

## Why

Neither exists at the moment an application is submitted — the process
hasn't reached any stage yet, let alone been reworked. Using them would
require moving the prediction point mid-process, which is a materially
different (and more complex) product than "flag risk on arrival."

Confirmed this reading against the spec's own dashboard mock-up (section
21, Page 4): its input form lists Loan Amount, Risk Score, Complexity,
Channel, Region, Priority, Current Queue — **no field for stage duration or
rework count**. That's independent evidence the spec's own intended
prediction point is arrival time, not mid-process.

## What replaced the gap

`current_queue_size`: a rolling count of applications submitted in the
trailing 24h before this one (`add_current_queue_size`, time-indexed
rolling window, `closed="left"` so it never counts itself). Fully derivable
from `application_date` alone — a real-time system-congestion signal, not
information about this specific application's own future.

## Leakage check

`check_no_leakage()` raises if any post-outcome column
(`final_status`, `total_processing_time_hours`, `sla_breached`, etc.) ever
ends up in the feature list — called in `train.py` before every training
run, not just at feature-build time, so a future refactor can't silently
reintroduce it.
