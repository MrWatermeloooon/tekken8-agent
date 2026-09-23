# Screen-only observations: design and first comparison

Date: 2026-09-23. Contract `screen-matchup-95-v1` (`--observation-mode screen`), defined in
`include/t8_v2/screen_observation.hpp`.

## Why

The `visual` contract was not what the live pipeline can see:

| Feature | `visual` in training | Live screen pipeline |
|---|---|---|
| Attack likelihood | exact move range and active frames | pixel motion above a threshold, times proximity |
| Motion | \|velocity\| + 0.08 while in a move | mean pixel difference per screen half |
| History: move ID, hit level, stance, repeat count | exact | unknown (−1) |
| Positions and distance | exact | noisy (one observed frame: 7.18 estimated vs 1.37 in-game) |

## Contract

- Base (13): HUD health with noise, measured positions with Gaussian noise, distance and velocity
  derived from those measured positions, exact health-drop hit events, and binary activity and
  attack-cue detections that flip at the lane's error rate.
- History (8 per step): observed opponent activity, observed attack cue, normalized position
  sigma, normalized event error, the opponent's visible-activity run length, outcome, distance,
  and opponent velocity. The opponent-action input is ignored, so no hidden move identity enters
  the observation.
- Uncertainty: each lane draws its position sigma from [0, 1.0] stage units and its error rate
  from [0, 0.30]. Both are also policy inputs. Live values come from
  `scripts/calibrate_screen_uncertainty.py`, and the live agent refuses a screen checkpoint
  without them.
- Tests: `tests/screen_observation_tests.cpp` checks that states differing only in the hidden
  move give identical observations (the `visual` control does leak), that noise-free features
  equal ground truth, that measured noise and flip rates match the configuration, and that the
  encoder ignores the opponent-action input and reports each lane's uncertainty.
  `tests/test_v2_live_policy.py` checks the live Python features and the calibration math.

## First comparison (one seed, 400 updates)

Both runs used identical options: 8,192 environments, horizon 128, `--curriculum-updates 400`
(scripted opponents only, no self-play), seed 4242, and 512 held-out episodes per evaluation.

| Update | Screen win rate (noisy eval) | Visual win rate (exact eval) |
|---:|---:|---:|
| 100 | 85.7% | 81.1% |
| 200 | 88.7% | 83.4% |
| 300 | 90.2% | 81.8% |
| 400 | 97.7% | 93.4% |

Cross-evaluation at 1,024 episodes. "Noisy" uses the default noise ranges; "zero noise" sets
both maxima to 0.

| Policy | Inputs | Win rate | Weakest style |
|---|---|---:|---:|
| Visual-trained, update 400 | visual | 92.4% | 68.0% |
| Visual-trained, update 400 | screen, noisy | 52.8% | 15.6% |
| Visual-trained, update 400 | screen, zero noise | 98.2% | 89.8% |
| Screen-trained, update 400 | screen, noisy | 98.2% | 88.3% |
| Screen-trained, update 400 | screen, zero noise | 93.0% | 56.2% |
| Overnight run, update 35,600 | visual | 89.6% | 68.8% |
| Overnight run, update 35,600 | screen, noisy | 45.2% | 5.5% |

Findings:

- Measurement noise, not the feature definitions, is what breaks visual-trained policies.
  Without noise they handle the screen features; with it they fall to 45–53%.
- The screen-trained policy stays at 93–98% across noise levels. Removing the hidden move
  information cost nothing in this comparison.
- Caveats: this is one seed and a simulated error model. Calibrating the live setup and
  repeating across seeds (`scripts/run_seed_matrix.ps1 -ExtraArgs '--observation-mode','screen'`)
  are the next steps before any claim about live play.
