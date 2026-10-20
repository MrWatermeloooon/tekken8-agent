from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("run_report", REPO_ROOT / "tools" / "run_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interval_math_matches_reference_values():
    report = _tool()
    # Wilson 95% interval for 8/10 (Newcombe 1998, Table I): 0.4902 to 0.9433.
    low, high = report.wilson(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4) and high == pytest.approx(0.9433, abs=1e-4)
    assert report.wilson(0, 0) == (0.0, 1.0)
    # Newcombe hybrid interval for 56/70 - 48/80 (Newcombe 1998b, example a): 0.0524 to 0.3339.
    difference, low, high = report.newcombe_difference(56, 70, 48, 80)
    assert difference == pytest.approx(0.2)
    assert low == pytest.approx(0.0524, abs=1e-3) and high == pytest.approx(0.3339, abs=1e-3)
    assert report.normalized_auc([0, 10, 20], [0.0, 1.0, 1.0]) == pytest.approx(0.75)
    assert report.normalized_auc([5], [1.0]) is None


def test_run_report_and_dashboard_from_run_artifacts(tmp_path):
    report = _tool()
    run = tmp_path / "run"
    run.mkdir()
    header = ("update,environment_steps,elapsed_seconds,episodes,training_opponent,latest_checkpoint_update,"
              "best_older_checkpoint_update,win_rate,p1_win_rate,p2_win_rate,draw_rate,worst_style_win_rate,"
              "mean_damage_dealt,mean_damage_taken,stochastic_win_rate,stochastic_p1_win_rate,"
              "stochastic_p2_win_rate,stochastic_draw_rate,stochastic_worst_style_win_rate,"
              "stochastic_mean_damage_dealt,stochastic_mean_damage_taken,guard_action\n")
    rows = [f"{update},{update * 100},{update},100,scripted,0,0,{rate},{rate},{rate},0.1,0.2,1,1,{rate},"
            f"{rate},{rate},0,0,1,1,none\n" for update, rate in ((10, 0.5), (20, 0.7), (30, 0.9))]
    (run / "evaluations.csv").write_text(header + "".join(rows), encoding="utf-8")
    (run / "evaluation_styles.csv").write_text(
        "update,policy,side,style,episodes,wins,losses,draws,win_rate\n"
        "30,deterministic,total,0,50,45,5,0,0.9\n30,deterministic,total,1,50,5,20,25,0.1\n"
        "30,stochastic,total,1,50,50,0,0,1\n", encoding="utf-8")
    (run / "metrics.jsonl").write_text("".join(
        json.dumps({"update": update, "environment_steps": update * 100, "elapsed_seconds": update / 10,
                    "approximate_kl": 0.01, "clip_fraction": 0.1, "entropy": 1.0, "decision_entropy": 2.0,
                    "value_loss": 0.1, "rollout_reward_mean": 0.01 * update}) + "\n"
        for update in (10, 20, 30)) + '{"update": 4', encoding="utf-8")  # last line still being written
    result = report.write_artifacts(run)
    assert result["last_update"] == 30 and result["latest"]["win_rate"] == 0.9
    assert result["latest"]["score"] == pytest.approx(0.95)
    assert result["auc_win_rate"] == pytest.approx(0.7)
    assert result["worst_style"]["style"] == "1" and result["worst_style"]["score"] == pytest.approx(0.35)
    assert result["final_window"]["episodes"] == 300
    page = (run / "dashboard.html").read_text(encoding="utf-8")
    assert page.count("<svg") == 7 and "Rollout reward" in page and "prefers-color-scheme: dark" in page
    summary = report.seed_summary([result, result])
    assert summary["final_window_win_rate"]["n"] == 2
