from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bradley_terry_ratings_follow_win_odds_and_draws(tmp_path):
    rank = _tool("rank_checkpoints")
    # A beats B 300:100 with no draws: odds 3:1 (slightly less after the virtual draw) = about 190 Elo.
    strengths = rank.fit([(0, 1, 300.0, 100.0, 0.0)], 2)
    elo = rank.to_elo(strengths, anchor=1)
    assert 180 < elo[0] - elo[1] < 191
    # All draws: equal ratings.
    even = rank.to_elo(rank.fit([(0, 1, 0.0, 0.0, 50.0)], 2), anchor=0)
    assert abs(even[1] - even[0]) < 1e-6
    # Transitive chain: A > B > C gives the same order.
    chain = rank.to_elo(rank.fit([(0, 1, 70, 30, 0), (1, 2, 70, 30, 0), (0, 2, 85, 15, 0)], 3), anchor=0)
    assert chain[0] > chain[1] > chain[2]
    table = tmp_path / "arena.csv"
    table.write_text("a,b,character_a,character_b,rounds,wins_a,wins_b,draws\n0,1,jun,jun,40,30,8,2\n", encoding="utf-8")
    (tmp_path / "arena.entrants.csv").write_text("index,checkpoint,character,provisional\n0,a.t8ppo,jun,1\n"
                                                 "1,b.t8ppo,jun,1\n", encoding="utf-8")
    result = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "rank_checkpoints.py"), str(table),
                             "--bootstraps", "50"], capture_output=True, text=True, check=True)
    assert "a.t8ppo" in result.stdout
    report = json.loads(table.with_suffix(".ratings.json").read_text(encoding="utf-8"))
    best = report["ratings"][0]
    assert best["entrant"] == "a.t8ppo" and best["ci95"][0] <= best["elo"] <= best["ci95"][1]
    # Reproducible: the same table and seed give the same report.
    subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "rank_checkpoints.py"), str(table),
                    "--bootstraps", "50"], capture_output=True, text=True, check=True)
    assert json.loads(table.with_suffix(".ratings.json").read_text(encoding="utf-8")) == report


def test_state_machine_skeletons_are_current_and_clean():
    generator = _tool("generate_state_machines")
    result = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "generate_state_machines.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    names = generator.stance_names({"stances": ["H", "ws", "CLK(Two", "CLK", "b+1:ORB", "ORB", "FUFT", "While", "hFC"],
                                    "moves": [{"command": "1,NSS.1", "recovery": "r20 NSS"}]})
    assert names == ["CLK", "ORB", "NSS"], "engine states, lying positions, and malformed tokens are dropped"
    yoshimitsu = yaml.safe_load((REPO_ROOT / "data/character_modules/yoshimitsu/state_machine.yaml").read_text(
        encoding="utf-8"))
    assert "NSS" in yoshimitsu["stances"] and "FUFT" not in yoshimitsu["stances"]
    assert all(rule["status"]["can_guard"] == "provisional" for rule in yoshimitsu["stances"].values())
    jun = (REPO_ROOT / "data/character_modules/jun/state_machine.yaml").read_text(encoding="utf-8")
    assert "generated skeleton" not in jun.splitlines()[0], "a reviewed file is never regenerated"
