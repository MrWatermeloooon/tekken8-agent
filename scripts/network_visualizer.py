"""Neural network visualizer: what the full-move policy looks like and what it is thinking.

    python scripts/network_visualizer.py [--run runs/train_all | --checkpoint PATH]
                                         [--character jun --opponent king] [--perception-delay N]

A desktop window (pygame), separate from the fight visualizer, drawn as a classic layered
network: input groups -> hidden layers (the 12 most important of 512 neurons each) -> the top
moves and the value. Live mode plays a fight through t8_v2_full_move_feed with the checkpoint on
both sides and recomputes the network on P1's actual input every frame; the bright path traces
the picked move back through its strongest contributors. Weights mode (W) shows the learned
weights instead.

Keys: W live/weights, Space pause, N next matchup, Esc quit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import struct
import subprocess
import sys
import threading

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""
STATE_FEATURES = 64
ROSTER = 42
POSTURES = ["standing", "crouching", "airborne", "grounded", "wakeup", "wall splat"]
LABELS = (["own health", "opp health", "own recoverable", "opp recoverable", "distance", "sidestep offset",
           "wall behind", "wall ahead"]
          + [f"own: {p}" for p in POSTURES] + [f"opp: {p}" for p in POSTURES]
          + ["own stun", "opp stun", "own blockstun", "opp blockstun", "own can act", "opp can act",
             "own Heat", "Heat time left", "Heat available", "own Rage", "Rage available", "opp Heat", "opp Rage",
             "own stance", "stance id", "opp stance", "gauge", "install", "own back turned", "opp back turned",
             "hits taken (combo)", "opp hits taken", "throw break window", "holding a throw", "Heat Dash ready",
             "own time down", "opp time down", "round time left", "own Ki charge", "opp Ki charge",
             "opp attacking", "opp hit: high", "opp hit: mid", "opp hit: low", "opp hit: throw",
             "opp frames to hit", "opp block advantage", "opp recovery left",
             "own attacking", "own move progress", "can tech", "own posture time", "opp posture time", "bias (1)"])
GROUPS = [(0, 4, "Health"), (4, 8, "Spacing"), (8, 20, "Posture"), (20, 26, "Stun and guard"),
          (26, 33, "Heat and Rage"), (33, 47, "State"), (47, 50, "Time"), (50, 58, "Opponent's move"),
          (58, 64, "Own move")]
ROSTER_SLUGS = ["kazuya", "jin", "king", "jun", "paul", "law", "jack-8", "lars", "xiaoyu", "nina", "leroy", "asuka",
                "lili", "bryan", "hwoarang", "claudio", "azucena", "raven", "leo", "steve", "kuma", "yoshimitsu",
                "shaheen", "dragunov", "feng", "panda", "lee", "alisa", "zafina", "devil-jin", "victor", "reina",
                "eddy", "lidia", "heihachi", "clive", "anna", "fahkumram", "armor-king", "miary-zo", "kunimitsu", "bob"]


# ---------------------------------------------------------------- checkpoint
class Network:
    """A .t8ppo checkpoint's layers (any depth) and observation normalizer, for numpy inference."""

    def __init__(self, path: Path) -> None:
        raw = path.read_bytes()
        header = struct.Struct("<8sIIIIQQ")
        contract = struct.Struct("<64s24s32s32sII")
        normalizer = struct.Struct("<IIQff")
        integrity = struct.Struct("<QQ")
        magic, version, observations, actions, hidden, self.step, parameters = header.unpack_from(raw)
        if magic != b"T8V2PPO\0" or version < 4:
            raise SystemExit(f"{path} is not a full-move checkpoint this viewer can read (version {version})")
        fields = contract.unpack_from(raw, header.size)
        self.feature_size = fields[4]
        offset = header.size + contract.size
        enabled, architecture, count, clip, epsilon = normalizer.unpack_from(raw, offset)
        offset += normalizer.size
        payload_bytes, _ = integrity.unpack_from(raw, offset)
        values = np.frombuffer(raw[offset + integrity.size:], dtype="<f4")
        self.shared, self.heads = (2, 0) if architecture == 0 else (architecture & 0xFF, (architecture >> 8) & 0xFF)
        outputs = (self.feature_size or actions) + 1
        shapes = [(hidden, observations if index == 0 else hidden) for index in range(self.shared)]
        if self.heads:
            shapes += [(hidden, hidden)] * self.heads * 2 + [(outputs - 1, hidden), (1, hidden)]
        else:
            shapes += [(outputs, hidden)]
        position = 0
        self.layers: list[tuple[np.ndarray, np.ndarray]] = []
        for out, inp in shapes:
            weights = values[position:position + out * inp].reshape(out, inp)
            position += out * inp
            bias = values[position:position + out]
            position += out
            self.layers.append((weights, bias))
        if position != parameters:
            raise SystemExit(f"{path}: unexpected layout ({position} parameters read, header says {parameters})")
        stats = values[position * 3:]
        self.mean = stats[:observations] if enabled and count else None
        self.inv_std = 1.0 / np.sqrt(stats[observations:2 * observations] + epsilon) if enabled and count else None
        self.clip = clip
        self.hidden = hidden
        self.observations = observations
        self.outputs = outputs

    @property
    def trunk(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.layers[:self.shared]

    def forward(self, observation: np.ndarray) -> dict:
        """Activations of every layer, the query, and the value."""
        x = observation.astype(np.float32)
        if self.mean is not None:
            x = np.clip((x - self.mean) * self.inv_std, -self.clip, self.clip)
        result = {"input": x, "trunk": [], "policy_head": [], "value_head": []}
        for weights, bias in self.trunk:
            x = np.tanh(weights @ x + bias)
            result["trunk"].append(x)
        if not self.heads:
            weights, bias = self.layers[self.shared]
            out = weights @ x + bias
            result["query"], result["value"] = out[:-1], float(out[-1])
            return result
        head_layers = self.layers[self.shared:self.shared + 2 * self.heads]
        p, v = x, x
        for weights, bias in head_layers[:self.heads]:
            p = np.tanh(weights @ p + bias)
            result["policy_head"].append(p)
        for weights, bias in head_layers[self.heads:]:
            v = np.tanh(weights @ v + bias)
            result["value_head"].append(v)
        (pw, pb), (vw, vb) = self.layers[-2], self.layers[-1]
        result["query"], result["value"] = pw @ p + pb, float((vw @ v + vb)[0])
        return result


def newest(directory: Path) -> Path | None:
    found = [p for p in directory.glob("update_*.t8ppo") if p.with_suffix(".t8fmstate").exists()]
    return max(found, key=lambda p: int(p.stem.split("_")[1])) if found else None


def trained_characters(checkpoint: Path) -> list[str]:
    data = checkpoint.with_suffix(".t8fmstate").read_bytes()
    offset, strings = 20, []
    for _ in range(3):
        length = int.from_bytes(data[offset:offset + 8], "little")
        strings.append(data[offset + 8:offset + 8 + length].decode("utf-8"))
        offset += 8 + length
    return [slug for slug in strings[2].split(",") if slug]


def perception_of(checkpoint: Path) -> int:
    """The --perception-delay the checkpoint's run trained with (run.json identity), else 0."""
    for run_json in (checkpoint.parent.parent / "run.json", checkpoint.parent / "run.json"):
        if run_json.exists():
            match = re.search(r"perception=(\d+)", run_json.read_text(encoding="utf-8"))
            return int(match.group(1)) if match else 0
    return 0


def load_move_names() -> dict[str, str]:
    names = {}
    with (REPO / "data" / "generated" / "full_combat_bindings.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            names[row["stable_id"]] = f"{row['command']}  {row['name']}".strip()
    return names


class Feed:
    def __init__(self, command: list[str]) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                        encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
        self.latest: queue.Queue = queue.Queue(maxsize=4)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if not line.startswith("{"):
                continue
            if self.latest.full():
                try:
                    self.latest.get_nowait()  # only the newest frames matter for display
                except queue.Empty:
                    pass
            self.latest.put(line)

    def close(self) -> None:
        self.process.kill()
        self.process.wait()


# ---------------------------------------------------------------- the app
SHOWN = 12  # neurons drawn per hidden layer (the most important ones right now)
INPUT_NODES = GROUPS + [(STATE_FEATURES, STATE_FEATURES + ROSTER, "Who I am"),
                        (STATE_FEATURES + ROSTER, STATE_FEATURES + 2 * ROSTER, "Who I'm facing")]
# --memory checkpoints (238 inputs) add each fighter's last 4 moves and habits after those.
MEMORY_START, MEMORY_SLOTS, MEMORY_FIGHTER = STATE_FEATURES + 2 * ROSTER, 40, 45
MEMORY_NODES = [(MEMORY_START, MEMORY_START + MEMORY_SLOTS, "My last moves"),
                (MEMORY_START + MEMORY_SLOTS, MEMORY_START + MEMORY_FIGHTER, "My habits"),
                (MEMORY_START + MEMORY_FIGHTER, MEMORY_START + MEMORY_FIGHTER + MEMORY_SLOTS, "Opponent's last moves"),
                (MEMORY_START + MEMORY_FIGHTER + MEMORY_SLOTS, MEMORY_START + 2 * MEMORY_FIGHTER, "Opponent's habits")]
BACKGROUND = (12, 13, 19)
POSITIVE = (232, 98, 108)
NEGATIVE = (112, 132, 228)


def load_move_features() -> dict[tuple[str, str], np.ndarray]:
    features = {}
    with (REPO / "data" / "generated" / "full_move_candidates.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            columns = [key for key in row if key.startswith("feature_")]
            features[(row["character"], row["stable_id"])] = np.array([float(row[k]) for k in columns], np.float32)
    return features


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        import pygame
        self.pg = pygame
        pygame.init()
        self.args = args
        self.screen = pygame.display.set_mode((1500, 900), pygame.RESIZABLE)
        pygame.display.set_caption("Tekken 8 Agent - Neural Network")
        font = lambda size, bold=False: pygame.font.SysFont("bahnschrift,segoeui,arial", size, bold=bold)
        self.fonts = {"small": font(14), "label": font(15, True), "title": font(22, True)}
        checkpoint = args.checkpoint or self.newest_champion()
        if checkpoint is None:
            raise SystemExit("no checkpoint found")
        self.moves = load_move_names()
        self.features = load_move_features()
        self.random = random.Random(args.seed)
        self.next_check = 0
        self.notice, self.notice_until = "", 0
        self.live, self.paused = not args.weights, False
        self.feed: Feed | None = None
        self.frame: dict | None = None
        self.activations: dict | None = None
        self.top: list = []  # P1's ranked choices [(stable id, probability)] from the feed
        self.chosen = ""
        self.matchup = None
        self.load(checkpoint)

    def newest_champion(self) -> Path | None:
        return newest(self.args.run / "champion") or newest(self.args.run / "checkpoints")

    def load(self, checkpoint: Path) -> None:
        """Load a checkpoint (any depth) and start a fight with it; keeps the matchup on a reload."""
        self.net = net = Network(checkpoint)
        self.checkpoint = checkpoint
        args = self.args
        self.delay = args.perception_delay if args.perception_delay is not None else perception_of(checkpoint)
        self.roster = trained_characters(checkpoint)
        memory = net.observations in (MEMORY_START + 2 * MEMORY_FIGHTER, MEMORY_START + 2 * MEMORY_FIGHTER + 3)
        noise = net.observations in (MEMORY_START + 3, MEMORY_START + 2 * MEMORY_FIGHTER + 3)
        noise_start = MEMORY_START + (2 * MEMORY_FIGHTER if memory else 0)
        self.input_nodes = (INPUT_NODES + (MEMORY_NODES if memory else [])
                            + ([(noise_start, noise_start + 3, "Noise levels")] if noise else []))
        self.activations = None
        self.hidden_layers = net.layers[:net.shared] + net.layers[net.shared:net.shared + net.heads]
        query_layer = net.layers[-2] if net.heads else net.layers[net.shared]
        self.query_weights = query_layer[0][:net.feature_size]
        self.importance = [np.zeros(net.hidden, np.float32) for _ in self.hidden_layers]
        # Weights mode: a neuron matters as much as its outgoing weights are large.
        nexts = [w for w, _ in self.hidden_layers[1:]] + [self.query_weights]
        self.static_importance = [np.linalg.norm(w, axis=0) for w in nexts]
        self.start_match(keep_matchup=self.matchup is not None)

    def check_for_new_champion(self) -> None:
        """Every 10 s (unless --checkpoint pinned one): switch to a newer champion of the run."""
        now = self.pg.time.get_ticks()
        if self.args.checkpoint or now < self.next_check:
            return
        self.next_check = now + 10_000
        latest = self.newest_champion()
        if latest is None or latest == self.checkpoint:
            return
        try:
            self.load(latest)
        except (OSError, SystemExit, ValueError):
            return  # still being written; try again at the next check
        self.notice, self.notice_until = f"new champion loaded: {latest.stem}", now + 6_000

    # ------------------------------------------------------------ data
    def start_match(self, keep_matchup: bool = False) -> None:
        if self.feed:
            self.feed.close()
        if keep_matchup and self.matchup:
            p1, p2 = self.matchup
        else:
            p1 = self.args.character or self.random.choice(self.roster)
            p2 = self.args.opponent or self.random.choice(self.roster)
        feed = REPO / "build-gpu" / "Release" / f"t8_v2_full_move_feed{EXE}"
        newer = REPO / "build-gpu2" / "Release" / feed.name  # memory-capable build while build-gpu is busy
        if self.net.observations > MEMORY_START and newer.exists():  # memory or noise inputs
            feed = newer
        # The feed plays with the run's own perception delay and hold frames (from run.json).
        command = [str(feed), "--character", p1, "--opponent", p2, "--checkpoint", str(self.checkpoint),
                   "--opponent-checkpoint", str(self.checkpoint), "--sample", "--emit-observation",
                   "--interval-ms", "33", "--seed", str(self.random.randrange(1 << 30)), "--allow-provisional"]
        if self.args.perception_delay is not None:
            command += ["--perception-delay", str(self.args.perception_delay)]
        self.feed = Feed(command)
        self.matchup = (p1, p2)
        self.top, self.chosen = [], ""
        self.decision, self.check_error = -1, None

    def poll(self) -> None:
        if not self.feed or self.paused:
            return
        line = None
        try:
            while True:
                line = self.feed.latest.get_nowait()
        except queue.Empty:
            pass
        if not line:
            return
        self.frame = json.loads(line)
        # Only a new decision changes the network: between decisions the move is held and nothing is computed.
        if self.frame.get("decision", 0) == self.decision or not self.frame.get("decision"):
            return
        self.decision = self.frame["decision"]
        choice = self.frame.get("choices", [{}])[0]
        self.top = [(move, float(p)) for move, p in choice.get("top", [])]
        self.chosen = choice.get("last", "")
        observation = np.array(self.frame.get("observation", []), dtype=np.float32)
        if observation.size == self.net.observations:
            self.activations = self.net.forward(observation)
            self.check_error = self.check_against_real_network()
            self.update_importance()

    def check_against_real_network(self) -> float | None:
        """Largest gap between this viewer's move probabilities and the real (CUDA) network's, among the
        shown moves: both softmax over the same legal set, so log(p_i / p_0) must equal score_i - score_0."""
        moves = [(m, p) for m, p in self.top if p > 1e-6 and (self.matchup[0], m) in self.features]
        if len(moves) < 2:
            return None
        query = self.activations["query"]
        scores = np.array([self.features[(self.matchup[0], m)] @ query for m, _ in moves]) / math.sqrt(self.net.feature_size)
        mine = np.exp(scores - scores[0])
        real = np.array([p for _, p in moves]) / moves[0][1]
        return float(np.max(np.abs(mine / mine.sum() - real / real.sum())))

    def hidden_values(self) -> list[np.ndarray]:
        acts = self.activations
        return acts["trunk"] + acts["policy_head"]

    def move_weights(self) -> np.ndarray:
        """How much each last-layer neuron adds to each shown move's score (moves x hidden)."""
        rows = [self.features.get((self.matchup[0], move), np.zeros(self.net.feature_size, np.float32))
                for move, _ in self.top]
        if not rows:
            return np.zeros((0, self.net.hidden), np.float32)
        return np.stack(rows) @ self.query_weights / math.sqrt(self.net.feature_size)

    def update_importance(self) -> None:
        """Backtrack from the move scores: how much each neuron contributes to what is picked now."""
        values = self.hidden_values()
        weights = self.move_weights()
        if len(weights):
            probability = np.array([p for _, p in self.top], np.float32)
            score = (probability[:, None] * np.abs(weights * values[-1][None, :])).sum(0)
        else:
            score = np.abs(values[-1])
        scores = [score]
        for column in range(len(values) - 1, 0, -1):
            matrix = self.hidden_layers[column][0]
            scores.insert(0, (scores[0][:, None] * np.abs(matrix * values[column - 1][None, :])).sum(0))
        for column, score in enumerate(scores):
            self.importance[column] = 0.85 * self.importance[column] + 0.15 * score / (score.max() or 1.0)

    def shown(self, column: int) -> list[int]:
        score = self.importance[column] if self.live and self.activations else self.static_importance[column]
        return sorted(int(i) for i in np.argsort(score)[-SHOWN:])

    # ------------------------------------------------------------ drawing
    def text(self, font, text, color, pos, anchor="topleft"):
        surface = self.fonts[font].render(text, True, color)
        self.screen.blit(surface, surface.get_rect(**{anchor: pos}))

    def column_positions(self, x: float, count: int, top: int, bottom: int) -> list[tuple[int, int]]:
        spacing = min(58.0, (bottom - top) / max(1, count - 1))
        y0 = (top + bottom) / 2 - spacing * (count - 1) / 2
        return [(int(x), int(y0 + spacing * i)) for i in range(count)]

    def node(self, pos, value, radius, highlight=False):
        pg = self.pg
        tint = POSITIVE if value >= 0 else NEGATIVE
        fill = mix((58, 34, 42) if value >= 0 else (34, 38, 64), tint, abs(value))
        if highlight:
            glow = pg.Surface((radius * 6, radius * 6), pg.SRCALPHA)
            for ring in range(3, 0, -1):
                pg.draw.circle(glow, (255, 170, 180, 34 * (4 - ring)), (radius * 3, radius * 3), radius + ring * 5)
            self.screen.blit(glow, (pos[0] - radius * 3, pos[1] - radius * 3))
            fill = mix(fill, (255, 214, 218), 0.6)
        pg.draw.circle(self.screen, fill, pos, radius)
        pg.draw.circle(self.screen, mix(fill, (255, 255, 255), 0.35), pos, radius, 2)

    def draw(self) -> None:
        pg = self.pg
        w, h = self.screen.get_size()
        net = self.net
        self.screen.fill(BACKGROUND)
        live = self.live and self.activations is not None
        top_y, bottom_y = 132, h - 90
        x_in, x_out = 270, w - 360
        columns = len(self.hidden_layers)
        xs = [x_in + (x_out - x_in) * (i + 1) / (columns + 1) for i in range(columns)]

        # Nodes and their values.
        observation = self.activations["input"] if live else None
        inputs = self.column_positions(x_in, len(self.input_nodes), top_y, bottom_y)
        input_values = [float(np.abs(observation[s:e]).max() / 2.0) if live else 0.0 for s, e, _ in self.input_nodes]
        shown = [self.shown(c) for c in range(columns)]
        hidden = [self.column_positions(xs[c], SHOWN, top_y, bottom_y) for c in range(columns)]
        values = self.hidden_values() if live else [np.tanh(b) for _, b in self.hidden_layers]
        moves = self.top
        out_count = max(1, len(moves))
        outputs = self.column_positions(x_out, out_count, top_y + 40, bottom_y - 140)

        # Connection strengths per stage: contribution (weight x input) live, the weight itself otherwise.
        stages = []
        first = self.hidden_layers[0][0]
        stage = np.zeros((SHOWN, len(self.input_nodes)), np.float32)
        for g, (s, e, _) in enumerate(self.input_nodes):
            block = first[np.array(shown[0])][:, s:e]
            stage[:, g] = (block * observation[None, s:e]).sum(1) if live else block.mean(1) * math.sqrt(e - s)
        stages.append((stage, inputs, hidden[0]))
        for c in range(1, columns):
            matrix = self.hidden_layers[c][0][np.ix_(shown[c], shown[c - 1])]
            if live:
                matrix = matrix * values[c - 1][np.array(shown[c - 1])][None, :]
            stages.append((matrix, hidden[c - 1], hidden[c]))
        move_matrix = self.move_weights()[:, shown[-1]]
        if live and len(move_matrix):
            move_matrix = move_matrix * values[-1][np.array(shown[-1])][None, :]
        if len(moves):
            stages.append((move_matrix, hidden[-1], outputs))

        # The highlighted path: the picked move back through its strongest contributors.
        path_nodes: list[tuple[int, int]] = []
        highlighted: set[tuple[int, int]] = set()
        if len(moves):
            ids = [m for m, _ in moves]
            target = ids.index(self.chosen) if self.chosen in ids else 0
            chain = [outputs[target]]
            for matrix, sources, _ in reversed(stages):
                source = int(np.argmax(np.abs(matrix[target])))
                chain.append(sources[source])
                target = source
            path_nodes = chain[::-1]
            highlighted = set(path_nodes)

        # Connections: every node to every node of the next layer, faint; the path on top, bright.
        overlay = pg.Surface((w, h), pg.SRCALPHA)
        for matrix, sources, targets in stages:
            scale = float(np.abs(matrix).max()) or 1.0
            for t, target in enumerate(targets):
                for s, source in enumerate(sources):
                    strength = abs(float(matrix[t, s])) / scale
                    color = POSITIVE if matrix[t, s] >= 0 else NEGATIVE
                    pg.draw.aaline(overlay, (*mix((150, 70, 86), color, strength), int(48 + 170 * strength ** 1.5)), source, target)
        for a, b in zip(path_nodes, path_nodes[1:]):
            pg.draw.line(overlay, (255, 150, 165, 70), a, b, 7)
            pg.draw.line(overlay, (255, 205, 212, 235), a, b, 3)
        self.screen.blit(overlay, (0, 0))
        if len(path_nodes) > 1 and not self.paused:
            phase = (pg.time.get_ticks() / 900.0) % 1.0 * (len(path_nodes) - 1)
            a, b = path_nodes[int(phase)], path_nodes[min(int(phase) + 1, len(path_nodes) - 1)]
            f = phase - int(phase)
            pulse = (int(a[0] + (b[0] - a[0]) * f), int(a[1] + (b[1] - a[1]) * f))
            glow = pg.Surface((40, 40), pg.SRCALPHA)
            for r, alpha in ((16, 40), (10, 90), (5, 255)):
                pg.draw.circle(glow, (255, 235, 238, alpha), (20, 20), r)
            self.screen.blit(glow, (pulse[0] - 20, pulse[1] - 20))

        # Nodes and labels.
        for (s, e, name), pos, value in zip(self.input_nodes, inputs, input_values):
            self.node(pos, value, 12, pos in highlighted)
            label = name
            if live and STATE_FEATURES <= s < MEMORY_START:
                raw = np.array(self.frame["observation"][s:e])
                label = f"{name}: {ROSTER_SLUGS[int(np.argmax(raw))]}"
            self.text("small", label, (205, 200, 212), (pos[0] - 22, pos[1]), "midright")
        for c in range(columns):
            for index, pos in zip(shown[c], hidden[c]):
                self.node(pos, float(values[c][index]), 11, pos in highlighted)
        for (move, probability), pos in zip(moves, outputs):
            self.node(pos, probability * 2, 14, pos in highlighted)
            name = self.moves.get(move, move.replace("universal:", "").replace("_", " "))
            self.text("small", f"{probability:4.0%}  {name[:30]}", (235, 230, 238), (pos[0] + 24, pos[1]), "midleft")
        if not moves:
            self.text("small", "(the moves appear in live mode)", (140, 140, 160), (x_out, (top_y + bottom_y) // 2),
                      "midleft")
        value = float(self.activations["value"]) if live else 0.0
        value_pos = (x_out, bottom_y - 40)
        self.node(value_pos, value, 14)
        self.text("small", f"value {value:+.2f}  (how good it thinks this is)", (235, 230, 238),
                  (value_pos[0] + 24, value_pos[1]), "midleft")

        # Layer titles and header.
        titles = [(x_in, f"Input ({net.observations})")]
        for c in range(columns):
            kind = "Hidden" if c < net.shared else "Policy head"
            titles.append((xs[c], f"{kind} {c + 1 if c < net.shared else c - net.shared + 1}"))
        titles.append((x_out, "Output"))
        for x, title in titles:
            self.text("label", title, (225, 220, 230), (int(x), h - 62), "midtop")
        for c in range(columns):
            self.text("small", f"{SHOWN} of {net.hidden} shown", (140, 138, 155), (int(xs[c]), h - 40), "midtop")
        self.text("small", "top moves", (140, 138, 155), (x_out, h - 40), "midtop")
        params = sum(wt.size + b.size for wt, b in net.layers)
        self.text("title", f"{self.checkpoint.parent.parent.name} / {self.checkpoint.stem}   ·   {params:,} parameters",
                  (232, 228, 238), (20, 14))
        if self.live and self.frame:
            f0, f1 = self.frame["fighters"]
            status = (f"LIVE   {self.matchup[0]} {f0['health']:.0f} HP  vs  {self.matchup[1]} {f1['health']:.0f} HP"
                      f"   ·   round {self.frame['round'] + 1}   ·   decision #{self.decision}"
                      f"   ·   bright path = what drives the move it picked")
            if self.check_error is not None:
                ok = self.check_error < 0.01
                self.text("small", ("matches the real network" if ok else "DOES NOT match the real network")
                          + f" (probability gap {self.check_error:.4f})", (120, 210, 140) if ok else (255, 110, 110),
                          (20, 94))
        else:
            status = "WEIGHTS   the neurons with the largest outgoing weights; lines are the weights themselves"
        self.text("small", status + ("   ·   PAUSED" if self.paused else ""), (165, 160, 180), (20, 46))
        if self.pg.time.get_ticks() < self.notice_until:
            self.text("label", self.notice, (255, 205, 120), (w - 20, 18), "topright")
        elif not self.args.checkpoint:
            self.text("small", "follows the newest champion", (120, 118, 135), (w - 20, 22), "topright")
        self.text("small", "W live/weights   Space pause   N next matchup   Esc quit", (120, 118, 135), (20, 70))

    def run(self) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    running = False
                elif event.type == pg.KEYDOWN and event.key == pg.K_w:
                    self.live = not self.live
                elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    self.paused = not self.paused
                elif event.type == pg.KEYDOWN and event.key == pg.K_n:
                    self.start_match()
            self.check_for_new_champion()
            self.poll()
            self.draw()
            pg.display.flip()
            clock.tick(30)
        if self.feed:
            self.feed.close()
        pg.quit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, default=REPO / "runs" / "train_all")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--character")
    parser.add_argument("--opponent")
    parser.add_argument("--perception-delay", type=int, default=None,
                        help="default: what the checkpoint's run trained with (run.json)")
    parser.add_argument("--weights", action="store_true", help="start in weights mode (W toggles)")
    parser.add_argument("--seed", type=int, default=random.randrange(1 << 30))
    parser.add_argument("--screenshot", type=Path, help="render a few seconds without a window, save a PNG, exit")
    args = parser.parse_args()
    if not args.run.is_absolute():
        args.run = REPO / args.run
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        app = App(args)
        import time
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and app.activations is None:
            app.poll()
            time.sleep(0.05)
        for _ in range(30):
            app.poll()
            time.sleep(0.033)
        app.draw()
        app.pg.image.save(app.screen, str(args.screenshot))
        app.feed.close()
        print(f"saved {args.screenshot}")
        return 0
    App(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
