# Tekken 8 Agent Implementation Plan

This plan keeps only the Tekken 8 part of `RL Speedrun Roadmap.pdf`.

## Goal

Build a local Tekken 8 self-play agent that can run as Player 1 through a
virtual controller, then let a friend play Player 2 in a local match.

Locked choices:

- Usage is offline local training and local versus only.
- Player 1 target character is Jun Kazama.
- Main training backend is a fast surrogate simulator.
- Tekken 8 state extraction starts with computer vision.
- DIAMBRA Arena is a reference/on-ramp, not the main training backend.

The project should be treated as three projects:

1. Build a stable fast Tekken-lite simulator.
2. Train and evaluate self-play policies on top of that simulator.
3. Calibrate and validate against DIAMBRA and real Tekken 8.

Training can start in the simulator before state extraction and input injection
are complete.

## Phase 0 - Repository And Safety Boundaries

Done criteria:

- All code lives under `D:\tekken 8`.
- The repo clearly states offline/local-only usage.
- No online play, matchmaking, anti-cheat bypassing, or stealth behavior.
- Game-specific addresses, offsets, and capture settings live in config files,
  not hardcoded deep inside training code.

## Phase 1 - State Extraction

This is now a calibration and validation phase, not the blocker for first
training. The simulator should already be able to run episodes before real
Tekken 8 state extraction works.

Target signals:

- Player 1 health.
- Player 2 health.
- Player 1 position.
- Player 2 position.
- Round timer.
- Round/match status.
- Optional later: move id, recovery state, blockstun/hitstun, distance, facing.

Preferred order:

1. Start with a backend interface and a mock backend.
2. Add a computer-vision backend for health bars and rough positions.
3. Add a memory-reading backend only for offline local testing if stable offsets
   can be found.

Why CV first may be useful:

- It can survive game patches better.
- It does not block the rest of the environment work.
- It gives us a fallback if memory offsets shift.

Why memory reading is still valuable:

- Lower latency when it works.
- Cleaner health/position/frame-state signals.
- Better reward and termination accuracy.

Done criteria:

- A script prints live state values at 10+ Hz during a real local match.
- Health changes correctly after hits.
- Positions change correctly when both players move.
- End-of-round is detected reliably.

## Phase 2 - Action Injection

Backend:

- Use `vgamepad` with a virtual Xbox 360 controller.
- Keep action injection behind `InputBackend` so we can test without touching the
  real game.

First action set:

- Neutral.
- Walk forward.
- Walk backward.
- Crouch.
- Jump.
- Left punch.
- Right punch.
- Left kick.
- Right kick.
- Simple combined actions after single-button actions work.

Done criteria:

- A script can move Player 1 in training mode.
- A script can perform each attack on command.
- Actions are rate-limited so held buttons and taps behave predictably.

## Phase 3 - Environment Loop

There are two environment loops:

- Simulator environment: fast, parallel, deterministic when seeded.
- Real Tekken 8 environment: slower, computer-vision based, used for validation
  and final local play.

Both should expose:

- `reset() -> Observation`
- `step(Action) -> StepResult`
- `close()`

Reward v1:

- Positive reward for damage dealt.
- Negative reward for damage taken.
- Large bonus for winning a round.
- Large penalty for losing a round.
- Small idle penalty when both distance and action state show no engagement.

Done criteria:

- Simulator environment can run hundreds of episodes without Tekken.
- Real environment can run one full local round and produce a reward trace.
- Episode logs include state, action, reward, and termination reason.

## Phase 4 - Self-Play

Use a checkpoint pool:

- Always keep recent checkpoints.
- Keep some older checkpoints.
- Sample mostly recent opponents, with occasional older opponents.
- Evaluate against a fixed ladder of checkpoints so progress is measurable.

Initial policy:

- Start with simple discrete actions.
- PPO is a reasonable first algorithm because the action space starts discrete.
- Consider recurrent policies later because fighting games have partial
  observability and timing matters.

Done criteria:

- Agent trains against simulator opponents.
- Checkpoints save and load.
- Evaluation reports win rate, damage differential, and idle rate.

## Phase 4A - Simulator Calibration

The simulator should be improved by comparing it to real Tekken 8 observations:

- Movement speed and spacing.
- Wall distance behavior.
- Health/damage scaling for Jun moves.
- Startup/active/recovery timing for a small Jun move list.
- Which attacks beat crouch/block/whiff at common ranges.

Done criteria:

- Each move in the simulator has an evidence note or TODO.
- Simulator policies that look good are tested against the real game.
- Real-game failures are converted into simulator fixes.

## Phase 5 - Live Play

Done criteria:

- Load a checkpoint.
- Run policy inference in real time.
- Send actions to Player 1.
- Friend plays Player 2 locally.
- Log the match for later reward/debug tuning.

## Immediate Questions

1. Which Jun moves should be added next?
2. Should first training use PPO or a simpler custom self-play baseline?
3. Which Tekken 8 measurement should calibrate the simulator first?
