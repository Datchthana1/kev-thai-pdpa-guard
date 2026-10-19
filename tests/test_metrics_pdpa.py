"""Unit tests for kev_qlora.metrics_pdpa (hand-computed cases) and an end-to-end run of
scripts/evaluate.py (EVAL_FAKE=1) followed by scripts/compare.py.

Run from the repo root:  .venv/Scripts/python.exe -m pytest tests -q
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kev_qlora import metrics_pdpa as m  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "tiny_test.jsonl"
EVALUATE = REPO / "scripts" / "evaluate.py"
COMPARE = REPO / "scripts" / "compare.py"


def approx(x):
    return pytest.approx(x, abs=1e-9)


# --------------------------------------------------------------------------- binary / F1
def test_binary_prf_hand_case():
    # y_true: T T F F T ; y_pred: T F F T T -> tp=2 (0,4), fn=1 (1), fp=1 (3), tn=1 (2)
    r = m.binary_prf([True, True, False, False, True], [True, False, False, True, True])
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 1)
    assert r["precision"] == approx(2 / 3)
    assert r["recall"] == approx(2 / 3)
    assert r["f1"] == approx(2 / 3)
    assert r["accuracy"] == approx(3 / 5)


def test_binary_prf_zero_division_is_zero():
    # no predicted positives and no true positives -> precision, recall, F1 all 0.0
    r = m.binary_prf([True, True], [False, False])
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0
    assert r["f1"] == 0.0


def test_binary_prf_perfect():
    r = m.binary_prf([True, False, True], [True, False, True])
    assert r["f1"] == approx(1.0)


def test_macro_f1_choice_hand_case():
    # labels a,b,c. y_true=[a,a,b] y_pred=[a,b,b]
    # a: P=1 R=1/2 F1=2/3 ; b: P=1/2 R=1 F1=2/3 ; c: no support, zero_division -> 0
    val = m.macro_f1(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "c"])
    assert val == approx((2 / 3 + 2 / 3 + 0) / 3)
    per = m.per_class_f1(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "c"])
    assert per["a"] == approx(2 / 3)
    assert per["c"] == 0.0


# --------------------------------------------------------------------------- per-answer helpers
def test_noul_threshold_at_half():
    assert m.noul_pred(0.5) is True
    assert m.noul_pred(0.49999) is False
    assert m.noul_pred(0.9) is True


def test_argmax_and_expected_score():
    probs = {"0": 0.0, "1": 0.5, "2": 0.5, "3": 0.0, "4": 0.0}
    assert m.argmax_key(probs) == "1"  # tie -> first in iteration order
    assert m.expected_score(probs) == approx(1.5)
    assert m.expected_score({"0": 0.1, "3": 0.2, "4": 0.7}) == approx(0.0 + 0.6 + 2.8)


def test_label_key_and_is_correct():
    assert m.label_key("noul", True) == "true"
    assert m.label_key("score", 3) == "3"
    assert m.label_key("choice", "contact") == "contact"
    assert m.is_correct({"type": "noul", "pred": True, "label": True})
    assert not m.is_correct({"type": "score", "pred": 2, "label": 3})
    assert m.is_correct({"type": "choice", "pred": "none", "label": "none"})


# --------------------------------------------------------------------------- calibration
def test_brier_noul_hand_case():
    # sample 1: probs false=.2 true=.8, label True -> (.2)^2 + (.8-1)^2 = .08
    # sample 2: probs false=.9 true=.1, label False -> (.9-1)^2 + (.1)^2 = .02
    val = m.calibration_for_probs(
        [{"false": 0.2, "true": 0.8}, {"false": 0.9, "true": 0.1}], ["true", "false"], n_bins=10
    )["brier"]
    assert val == approx((0.08 + 0.02) / 2)


def test_ece_single_bin_hand_case_10_bins():
    # four predictions at confidence 0.9, accuracy 3/4 -> ECE = |0.75 - 0.9| = 0.15
    bins = m.reliability_bins([0.9, 0.9, 0.9, 0.9], [True, True, True, False], n_bins=10)
    assert sum(b["n"] for b in bins) == 4
    assert m.ece_from_bins(bins) == approx(0.15)


def test_ece_single_bin_hand_case_15_bins():
    bins = m.reliability_bins([0.9, 0.9, 0.9, 0.9], [True, True, True, False], n_bins=15)
    assert m.ece_from_bins(bins) == approx(0.15)


def test_ece_mixed_two_bins_hand_case():
    # bin A (conf 0.5): 2 items, acc 0.5 -> gap 0 ; bin B (conf 1.0): 2 items, acc 0.5 -> gap 0.5
    # ECE = 2/4*0 + 2/4*0.5 = 0.25 ; confidence 1.0 must land in the last bin
    bins = m.reliability_bins([0.5, 0.5, 1.0, 1.0], [True, False, True, False], n_bins=15)
    assert bins[-1]["n"] == 2
    assert m.ece_from_bins(bins) == approx(0.25)


def test_ece_empty_is_none():
    assert m.ece_from_bins(m.reliability_bins([], [], 15)) is None


# --------------------------------------------------------------------------- risk / latency
def test_risk_stats_hand_case():
    gold = [0, 3, 4, 2]
    argmax = [0, 2, 4, 2]
    expected = [0.1, 2.4, 4.0, 2.0]
    r = m.risk_stats(gold, argmax, expected)
    assert r["exact_acc"] == approx(3 / 4)
    assert r["within1_acc"] == approx(1.0)  # |3-2| = 1 counts as within one
    assert r["mae_argmax"] == approx(1 / 4)  # (0+1+0+0)/4
    assert r["mae_expected"] == approx((0.1 + 0.6) / 4)  # (0.1+0.6+0+0)/4 = 0.175


def test_latency_stats_hand_case():
    s = m.latency_stats([1, 2, 3, 4, 5])
    assert s["mean_s"] == approx(3.0)
    assert s["p50_s"] == approx(3.0)
    assert s["p95_s"] == approx(4.8)  # linear interpolation at index 3.8
    assert s["n"] == 5


def test_jsonable_handles_nan_and_numpy():
    import numpy as np

    out = m.jsonable({"a": float("nan"), "b": np.float64(0.5), "c": np.int64(3), "d": (1, 2)})
    assert out == {"a": None, "b": 0.5, "c": 3, "d": [1, 2]}


# --------------------------------------------------------------------------- compute_metrics
def _row(rid, genre, hn, gold_risk, sens_pred, sens_label, risk_pred, primary_pred, primary_label, latency):
    answers = {
        "is_sensitive": {"type": "noul", "probs": {"false": 1 - sens_pred, "true": sens_pred},
                         "label": sens_label, "pred": sens_pred >= 0.5},
        "risk_level": {"type": "score", "probs": {str(i): (1.0 if i == risk_pred else 0.0) for i in range(5)},
                       "label": gold_risk, "pred": risk_pred, "expected": float(risk_pred)},
        "primary_category": {"type": "choice", "probs": {"none": 0.0, "contact": 1.0},
                             "label": primary_label, "pred": primary_pred},
    }
    for a in answers.values():
        a["correct"] = m.is_correct(a)
    return {"id": rid, "genre": genre, "hard_negative": hn, "risk_label": gold_risk,
            "answers": answers, "latency_s": latency}


def test_compute_metrics_small_hand_case():
    rows = [
        _row("a", "chat", False, 2, 0.9, True, 2, "contact", "contact", 0.1),
        _row("b", "chat", True, 0, 0.7, False, 0, "none", "none", 0.2),  # false positive for is_sensitive
        _row("c", "news", True, 0, 0.2, False, 1, "none", "none", 0.3),
    ]
    met = m.compute_metrics(rows, choice_labels=["none", "contact"], n_bins=10)

    # is_sensitive: tp=1 fp=1 fn=0 -> P=1/2 R=1 F1=2/3
    assert met["is_sensitive"]["f1"] == approx(2 / 3)
    assert met["accuracy_by_qid"]["is_sensitive"]["accuracy"] == approx(2 / 3)  # a ok, b wrong, c ok -> 2/3
    assert met["accuracy_by_qid"]["risk_level"]["accuracy"] == approx(2 / 3)  # b predicted 0 = gold 0 ok; c predicted 1 vs 0 wrong; a ok
    # risk: gold [2,0,0], argmax [2,0,1] -> exact 2/3, within1 1.0, mae 1/3
    assert met["risk_level"]["exact_acc"] == approx(2 / 3)
    assert met["risk_level"]["within1_acc"] == approx(1.0)
    assert met["risk_level"]["mae_argmax"] == approx(1 / 3)
    assert met["risk_level"]["confusion"][2][2] == 1
    assert met["risk_level"]["confusion"][0][1] == 1
    # primary accuracy: a ok, b ok, c ok -> 1.0
    assert met["primary_category"]["accuracy"] == approx(1.0)
    # slices
    assert set(met["by_genre"]) == {"chat", "news"}
    assert met["by_genre"]["news"]["n"] == 1
    assert set(met["by_gold_risk"]) == {"0", "2"}
    assert set(met["by_hard_negative"]) == {"true", "false"}
    assert met["by_hard_negative"]["false"]["n"] == 1
    assert met["latency"]["mean_s"] == approx(0.2)
    assert set(met["calibration"]["by_type"]) == {"noul", "score", "choice"}
    assert set(met["calibration"]["by_qid"]) == {"is_sensitive", "risk_level", "primary_category"}


def test_compute_metrics_missing_genre_and_no_positives_do_not_crash():
    rows = [_row("a", None, False, 0, 0.1, False, 0, "none", "none", 0.1)]
    met = m.compute_metrics(rows, choice_labels=["none", "contact"], n_bins=10)
    assert list(met["by_genre"]) == ["unknown"]
    assert met["is_sensitive"]["f1"] == 0.0
    assert met["is_sensitive"]["precision"] == 0.0


# --------------------------------------------------------------------------- evaluate helpers
def _load_evaluate_module():
    spec = importlib.util.spec_from_file_location("evaluate_script", EVALUATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_public_questions_strip_label_and_meta():
    ev = _load_evaluate_module()
    rec = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    pub = ev.public_questions(rec["questions"])
    assert "meta" in rec  # the source record has meta; it must not reach the systems
    for qid, q in pub.items():
        assert set(q) <= {"type", "instructions", "criteria"}
        assert "label" not in q
    assert pub["risk_level"]["criteria"] == rec["questions"]["risk_level"]["criteria"]


def test_normalize_answer_rejects_bad_keys():
    ev = _load_evaluate_module()
    q = {"type": "noul", "instructions": "x", "label": True}
    with pytest.raises(ValueError):
        ev.normalize_answer("is_sensitive", q, {"probs": {"yes": 0.5, "no": 0.5}}, "r1")


# --------------------------------------------------------------------------- end to end
def _run(args, env_extra=None):
    env = dict(os.environ)
    env["EVAL_FAKE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, *args], cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8"
    )


def test_evaluate_and_compare_end_to_end(tmp_path):
    out_dir = tmp_path / "results"
    out_dir.mkdir()

    # three systems on the fixture (fake system), the last one with a missing genre slice when limited
    for sysname in ("kev-base", "openthai"):
        out = out_dir / f"eval_{sysname}.json"
        proc = _run([str(EVALUATE), "--system", sysname, "--test", str(FIXTURE), "--out", str(out)])
        assert proc.returncode == 0, proc.stderr
        for i in range(1, 7):
            assert f"rec={i}/6 id=" in proc.stdout
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["system"] == sysname
        assert data["n_records"] == 6
        assert len(data["per_question"]) == 6
        assert data["config"]["fake"] is True
        met = data["metrics"]
        assert met["accuracy_by_qid"]
        assert len(met["accuracy_by_qid"]) == 11
        assert met["latency"]["n"] == 6
        assert set(met["calibration"]["by_type"]) == {"noul", "score", "choice"}
        # identical question inputs: every system sees the same qids/types for each record
        first = data["per_question"][0]
        assert set(first["answers"]) == {
            "is_sensitive", "risk_level", "primary_category", "has_name", "has_contact", "has_address",
            "has_national_id", "has_financial", "has_credential", "has_health_or_special", "has_other_identifier",
        }
        assert (out_dir / f"eval_{sysname}.partial.jsonl").exists()

    # a resumed run reuses the partial file and does not call the system again
    out = out_dir / "eval_kev-base.json"
    proc = _run([str(EVALUATE), "--system", "kev-base", "--test", str(FIXTURE), "--out", str(out)])
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("resumed") == 6

    # one system with a limit (fewer records)
    lim = out_dir / "eval_kev-pdpa.json"
    proc = _run([str(EVALUATE), "--system", "kev-pdpa", "--run", "dummy_run", "--test", str(FIXTURE),
                 "--limit", "3", "--out", str(lim), "--fresh"])
    assert proc.returncode == 0, proc.stderr
    assert json.loads(lim.read_text(encoding="utf-8"))["n_records"] == 3

    # compare: three systems, then a single-system run (must also work)
    proc = _run([str(COMPARE), "--results-dir", str(out_dir)])
    assert proc.returncode == 0, proc.stderr
    md = (out_dir / "comparison.md").read_text(encoding="utf-8")
    assert "Headline metrics" in md and "By genre" in md and "By gold risk level" in md
    assert "kev-pdpa" in md and "openthai" in md
    for png in ("bar_headline.png", "bar_by_genre.png", "risk_confusion_kev-base.png",
                "reliability_kev-base.png", "risk_confusion_openthai.png", "reliability_openthai.png"):
        assert (out_dir / png).stat().st_size > 0, png

    single = tmp_path / "single"
    single.mkdir()
    (single / "eval_kev-base.json").write_text((out_dir / "eval_kev-base.json").read_text(encoding="utf-8"),
                                               encoding="utf-8")
    proc = _run([str(COMPARE), "--results-dir", str(single), "--systems", "kev-base"])
    assert proc.returncode == 0, proc.stderr
    assert (single / "comparison.md").exists()
    assert (single / "bar_headline.png").exists()


def test_compare_fails_cleanly_without_results(tmp_path):
    proc = _run([str(COMPARE), "--results-dir", str(tmp_path)])
    assert proc.returncode == 1
