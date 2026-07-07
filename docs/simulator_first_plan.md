# Simulator-First Plan

The fastest path is to train in a compact fighting-game simulator, not in the
real Tekken 8 process. Real Tekken 8 is too slow for early self-play and too
awkward for reward debugging. The simulator gives us parallel games, cheap
experiments, and a place to learn fundamentals before computer vision matters.

## Why This Beats Real-Time Training First

- Simulator episodes can run much faster than real time.
- Many simulator environments can run in parallel.
- Reward bugs are easier to inspect.
- Self-play checkpoint pools can be tested before Tekken 8 input/state plumbing.
- Real Tekken 8 becomes a calibration target instead of the first bottleneck.

## Current Simulator Scope

Implemented now:

- Health, round timer, win/loss termination.
- 2.5D spacing with walls and body separation.
- Decision frames for RL-style stepping.
- Startup, active, recovery, range, damage, pushback.
- High/mid/low/throw hit levels.
- High block and low block.
- Hitstun, blockstun, whiffs, launch flag.
- First Jun-style move table: jab, df1, f2, db3, hopkick, throw.

## Near-Term Additions

1. Add more Jun moves with rough frame-data placeholders.
2. Add action masks so busy fighters cannot start impossible actions.
3. Add scripted opponents: turtle, rushdown, whiff-punisher, low-spammer.
4. Add vector observations for RL.
5. Add parallel environment runner.
6. Add PPO or a simpler first self-play learner.
7. Save checkpoint pools and evaluate against old policies.

## Calibration Loop

When Tekken 8 CV is ready, use real matches to tune the simulator:

- Measure health bar changes for known moves.
- Estimate movement speed and common distances.
- Compare block/hit/whiff outcomes.
- Add wall behavior only when the basic open-stage model trains useful play.

The simulator does not need to be perfect. It needs to teach transferable
fundamentals: spacing, timing, punishment, blocking, risk, and round-winning.
