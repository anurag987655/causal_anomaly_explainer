# Publishability Checklist

## A. Protocol Integrity
- [ ] Headline metrics are leakage-safe (global threshold learned on validation only).
- [ ] Oracle diagnostics (label-aware thresholds) are clearly separated from headline metrics.
- [ ] Random split results reported.
- [ ] Time split results reported with effective split strategy logged.
- [ ] Cross-dataset benchmark results reported.

## B. Statistical Rigor
- [ ] All primary metrics include 95% bootstrap confidence intervals.
- [ ] At least 3 seeds or robust justification for fixed-seed usage.
- [ ] Ablation table included and discussed.
- [ ] Weak-family performance table included (WebAttack/Infiltration/Bot where available).

## C. Reproducibility
- [ ] One-command publishable run works from clean environment.
- [ ] `reproducibility_manifest.json` generated and archived.
- [ ] Input/output artifact paths documented.
- [ ] Requirements and environment details recorded.

## D. Writing Quality
- [ ] Claims are tied to explicit tables/figures.
- [ ] Limitations and threats to validity section is explicit.
- [ ] No claim exceeds measured evidence.
- [ ] Comparison against baseline is explicit and fair.

## E. Ethics and Deployment Safety
- [ ] Potential misuse and false-positive cost are discussed.
- [ ] Operational constraints and monitoring requirements are stated.
