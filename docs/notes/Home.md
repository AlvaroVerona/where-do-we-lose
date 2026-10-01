# Where do We Lose? — Vault

Notes vault for this repo, kept inside it so it's versioned with git instead
of living somewhere disconnected. It doesn't duplicate the project's own
documentation — `README.md` and `data/README.md` stay the single source of
truth for architecture, results, and the data-generation methodology. This
vault is for the things those docs aren't good at: decisions with their
*why* pulled out onto their own page, the bugs actually found and fixed
along the way, and open questions that don't belong to any one file.

## Map

- [[Decisions/Bottleneck Score Formula]] — weighted sum vs. the spec's illustrative product formula
- [[Decisions/Resource Utilization Method]] — why it's peer-relative, not hours-based
- [[Decisions/Prediction Point]] — why the SLA-breach model predicts at submission time, not mid-process
- [[Decisions/Optimization Cost Model]] — the analytical-proxy vs. real-simulation split
- [[Bugs Found & Fixed]] — all 8, across every phase, with root cause and fix
- [[Backlog/Open questions]] — flagged but not resolved

> [!info] Source of truth (outside this vault)
> These links point outside the `notes/` folder — Obsidian will open them, but won't track backlinks to them the way it does for notes inside the vault.
> - [README](../README.md) — architecture, methodology, results, screenshots
> - [Data generation methodology](../data/README.md) — every business rule and calibration decision behind the synthetic dataset
> - [Project specification](../../Project%20Specification%20—%20Process%20Bottleneck%20Detection%20&%20Optimization%20System.md) — the original spec this project was built against
