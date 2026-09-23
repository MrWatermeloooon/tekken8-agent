# Self-play held-out oscillation: diagnosis and audit

Run: `overnight_roster_visual_shaped_seed20260722` (32,768 environments, horizon 128, 35,600
updates, visual observations, shaped reward). Analysis date: 2026-09-23.

## Summary

The 97.3% / 30.5% swings are real changes in policy behavior, not evaluation noise. The policy
switches between two near-deterministic strategies:

| Regime | Checkpoints | Held-out win rate | Choice decisions spent on |
|---|---|---:|---|
| "f2 pressure" | 100, 12,000, 15,000, 26,000, 35,600 | 89.6% – 97.1% | f2 34–60%, dash forward 25–56% |
| "dash-forward spam" | 3,700, 6,100, 8,300, 18,100 | 33.4% – 57.5% | dash forward 74–83%, f2 2–10% |

The trainer runs self-play for 99.8% of updates, always as a Jun mirror against a narrow pool
(80% latest self, 20% one older checkpoint). Nothing in that pool reliably punishes dash spam.
The scripted held-out suite does, so held-out score collapses until self-play drifts back.

## Evidence

**Noise versus signal.** The median change between consecutive 256-episode evaluations is 2.3
points. Binomial noise at p≈0.8 is 2.5 points, so that small wobble is noise. The troughs are
different: they last thousands of updates (3,700–8,800 and 16,000–20,000), and the probes below
use 1,024 episodes (±1.4 points).

**Behavior probes.** The new `t8_v2_train --probe-checkpoint` mode runs the held-out suite with
behavior statistics. For the V2-format checkpoints of this run, first convert with
`scripts/convert_v2_checkpoint_to_v3.py --contract-from <checkpoint written by the current
trainer>`. The probe reproduced the logged update-35,600 evaluation exactly (88.28%
deterministic, 90.63% stochastic, same seed), which validates the checkpoint loading, temporal
encoding, and side-routing paths. Results at 1,024 episodes:

| Update | Win rate (P1 / P2) | Choice fraction | Entropy per choice | Top choices |
|---:|---|---:|---:|---|
| 100 | 92.9% (93.6 / 92.2) | 0.21 | 0.98–1.14 | f2 .36, dash_forward .18, sidewalk_right .14 |
| 3,700 | 49.8% (60.9 / 38.7) | 0.53–0.58 | 0.08–0.15 | dash_forward .74, walk_forward .10 |
| 6,100 | 33.4% (24.8 / 42.0) | 0.54–0.75 | 0.10–0.24 | dash_forward .80 |
| 8,300 | 35.3% (40.6 / 29.9) | 0.62–0.70 | 0.13–0.30 | dash_forward .75, walk_forward .14 |
| 12,000 | 97.1% (95.5 / 98.6) | 0.19–0.21 | 0.22–0.30 | f2 .38, dash_forward .36 |
| 15,000 | 96.1% (97.1 / 95.1) | 0.20–0.27 | 0.10–0.15 | dash_forward .55, f2 .34 |
| 18,100 | 57.5% (51.6 / 63.5) | 0.37–0.66 | 0.03–0.14 | dash_forward .83, f2 .10 |
| 26,000 | 90.6% (90.6 / 90.6) | 0.14 | 0.08–0.09 | f2 .59, dash_forward .35 |
| 35,600 | 89.6% (89.5 / 89.8) | 0.15 | 0.18 | f2 .52, dash_forward .25, sidewalk_left .10 |

"Choice" decisions are live frames with more than one legal action. The maximum entropy for 24
actions is ln 24 ≈ 3.18. Every checkpoint after update 100 is close to deterministic, and even
the best ones use mostly two actions. That matches the "only heavy attacks and moving forward"
behavior seen in live play, so that behavior is not only a screen-capture artifact.

**Troughs lose across the board.** The dash-spam checkpoints fall to 5–45% against held-out
styles 1 and 4, and to at most 80% against every style. The f2 checkpoints keep 82–100% on most
styles; style 6 is the weakest for every f2-regime checkpoint (52–92%).

**The logged `entropy` was misleading.** It averages over every sample, including forced frames
with one legal action (entropy 0). It therefore mostly measures how often the policy has a
choice, not how random it is when it does. A fresh network logs `entropy` 0.83, which equals a
0.26 choice fraction × 3.18. The rise to 1.9 late in the run was measured on self-play rollouts;
on held-out states the same checkpoint's per-choice entropy is 0.18. The trainer now logs
`decision_fraction` and `decision_entropy` separately.

**Shaped reward is not the driver.** The per-step shaping terms are −0.02 to −0.18, and dash
forward does avoid the whiff, idle, wall, and lateral penalties. Terminal and damage terms are
±40 to 320 per round, however, so shaping is at most a tie-breaker. The more likely cause is the
opponent distribution:

- Self-play began at update 101. `--curriculum-updates` was left at its default of 100, so
  stages 1–3 covered 75 updates (0.2% of the run).
- Self-play is always Jun against Jun (`jun_profile_index`); roster opponents never return.
- The pool holds two opponents. Latest-self trails by up to one checkpoint interval (100
  updates) and is often a copy of the same dash strategy. The single best-older checkpoint (100
  until update 11,800, then 15,000 for the rest of the run) is 20% of matches, which is not
  enough to stop the drift.

The mechanism (the dash strategy beating a recent copy of itself) is inferred from the data
above and has not been tested separately.

## Audit: latest-self loading, frozen history, side routing, reset

No correctness bug was found. Checked:

- **Checkpoint selection and loading.** Latest-self is the newest checkpoint older than the
  current update. Opponents load weights only (`load_optimizer_state=false`) and sample
  stochastically with their own seeds (+2 latest, +4 best-older).
- **Temporal history.** The opponent encoder receives the learner's previous actions. The
  learner encoder receives the opponent's previous actions, and both are recorded after
  encoding, so there is no one-step lookahead. `preview` does not advance state. Both encoders
  reset when switching between scripted play and self-play and on `terminated`, which includes
  timeouts; `truncated` is a subset. A `valid` flag blocks the previous episode's last action
  from entering a new episode. When latest-self changes mid-episode, the history carries over
  unchanged, which is harmless because it records observations rather than policy state. The
  policy has no recurrent state (see the PPO notes on frame stacking).
- **Side routing.** Training (`training_router.cu`), character assignment (`gpu_sim.cu`,
  `learner_player = 0`), and rewards all use the same rule: `(lane / 8) & 1` selects P2. The
  best-older mixture (`lane % 5 == 0`) sends exactly 20% of each side's lanes to the older
  opponent.
- **Exact resume.** The self-play exact-resume smoke test (`t8_v2_exact_resume`) passes, and
  the reconstructed evaluation matched the ledger exactly.

Fixed during the audit:

- Checkpoint selection re-parsed the whole metrics ledger every update (20.7 MB at update
  35,600). It now reads only newly appended bytes (`EvaluationLedger`).
- Both opponent checkpoints were reloaded and checksummed every update even when unchanged. They
  now reload only when the selection changes.

Design weaknesses (not bugs), tracked in the README:

- Best-older selection takes the highest single 256-episode score. After a lucky 97.3% at
  update 15,000, no later checkpoint could win by enough to replace it.
- The Jun-mirror-only, two-opponent pool described above.

## Why checkpoint 100 stayed the best-older opponent

Every checkpoint from 100 to 11,900 was re-probed at 1,024 episodes (±0.8 points). The probes
track the logged 256-episode scores closely (mean absolute difference 1.6 points, correlation
0.994).

- **Mostly legitimate.** From update 300 to 11,700, no checkpoint beat update 100's 92.9%. The
  run was in the dash-forward regime, and the best probe in that stretch was 89.1% (update
  1,000). Checkpoint 100 predates self-play; it was trained against scripted opponents.
- **One noise error.** Update 200 was actually stronger (95.3% against 92.9% at 1,024 episodes).
  It lost the logged comparison by a single game (92.97% against 93.36% at 256 episodes), so the
  strict maximum kept 100.
- **Replacement.** Update 11,800 (probe 95.0%, logged 95.7%) was the first checkpoint to
  replace it, as the ledger shows.

**Fix:** `--promotion-tie-band` (default 1 standard error) treats scores within one binomial
standard error of the best as tied and promotes the newest. Replayed on this ledger, it promotes
update 200 from update 400 onward, matches the old rule after that, and still does not replace
update 15,000. That checkpoint's 97.3% is not approached within the band later in the run.

## Regression guard, replayed on this run

With default thresholds, the guard would first have triggered at update 800. The held-out score
had stayed more than 15 points below checkpoint 100's 93.4% for three consecutive evaluations.
Under this self-play recipe it would roll back twice and then pause, which is the intended
signal that the recipe itself regresses. The trajectory after a rollback cannot be replayed from
the old ledger.

## Side-gap promotion threshold

`--promotion-max-side-gap` (default 0.10) allows an older checkpoint to become best-older only if
its held-out |P1 − P2| win-rate gap is at most the threshold. If no checkpoint qualifies, the most
balanced one is used, and `best_older_side_gap_fallback` is logged. A value of 1 disables the
filter. Resuming a trainer state from before version 7 requires `--promotion-max-side-gap 1`.

Replaying this run's ledger shows the threshold would not have changed any promotion: every
promoted checkpoint (100, 11,800, 12,000, 14,500, 15,000) had a gap of at most 3.9 points. The side
gap therefore did not cause the oscillation.

The 10 checkpoints after the 34,600 snapshot (34,700–35,600) had gaps of −6.2 to 0.0 points,
with P2 ahead or level in all of them, and all within 256-episode noise (sd ≈ 4.5 points for the
difference). At 1,024 episodes, update 35,600 measured 89.5% / 89.8%. The balance held across
later checkpoints of this seed. No other long seed exists yet, so it has not been checked across
seeds.
