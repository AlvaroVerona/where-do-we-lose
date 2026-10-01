# Open questions

Real gaps or considerations flagged live, not fixed (yet), not silently
ignored either.

## PM4Py is AGPL v3

Used in `notebooks/02_process_mining.ipynb` for an independent DFG-discovery
cross-check, per the spec's own tech stack recommendation. AGPL's copyleft
terms apply to network services — if the Streamlit dashboard is ever
deployed as a public-facing service (not just run locally), that license
needs a real look before shipping. Not an issue for local/portfolio use.

## Resource utilization doesn't capture absolute staffing tightness

See [[Decisions/Resource Utilization Method]]. The peer-relative-throughput
method is bounded and honest about what it measures, but it came back
fairly flat (~76–82%) across all six stages even though Credit Assessment
is genuinely the most understaffed relative to its workload. A
cross-department benchmark (e.g. hours-per-case estimated from
low-congestion days specifically, rather than a blanket percentile) could
recover that signal — not attempted, given the complexity/return tradeoff
for a portfolio-scale project.

## Optimization's employee cost is a uniform per-head figure

`resource_allocation.py`'s cost term treats every employee at every stage
as equally expensive. Real compensation varies by role/department/seniority
— plugging in real per-role cost data would make the optimizer's
cost/benefit tradeoff more realistic, and is exactly the kind of input a
real deployment would have and this synthetic project doesn't.

## Nested git repository

This project lives inside `~/`, which is itself a git repository (tracking
dotfiles). `git init` here created a second, independent repo scoped to
`process-bottleneck-optimization/` rather than nesting inside the home
repo's history — deliberate, since this project is meant to be published
on its own. Worth double-checking before pushing anywhere that the outer
home-directory repo doesn't have anything configured (e.g. a submodule
reference) that would conflict.

## `reports/figures/*.png` is gitignored except `.gitkeep`

`notebooks/02_process_mining.ipynb`'s `process_flow.png` (the networkx flow
diagram) regenerates on every notebook re-run but isn't committed —
intentional (generated artifact), but means a fresh clone that only reads
the repo (without executing the notebook) won't see that specific figure.
The dashboard screenshots in `reports/screenshots/` *are* committed and
cover the same ground for the README.
