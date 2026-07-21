# Archived Phase 0 result — superseded protocol

The earlier five-seed result evaluated against the same `scripted_v1` family used during training,
always started from one fixed state, and did not balance/report learner side. Both reward modes
saturated that in-distribution benchmark, so the old “no-go for redistribution” conclusion is not
promotion-quality evidence.

Those artifacts remain historical only. The replacement protocol uses:

- a distinct held-out V2 opponent suite;
- equal learner-as-P1 and learner-as-P2 episodes;
- fair equal-health timeout draws;
- deterministic randomized starting positions;
- per-style, per-side, timeout, stalemate, frame, and damage metrics;
- the deployable 13-feature visual observation contract by default;
- automated median/IQR and normalized learning-curve AUC across at least three seeds.

Run `scripts/run_phase0.ps1`; its generated report under `docs/` is the only current Phase 0 gate
artifact. The random-opponent pilot in `phase0_pilot_random.md` is also retained only as history.
