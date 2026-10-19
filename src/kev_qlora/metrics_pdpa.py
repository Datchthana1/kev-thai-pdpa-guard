"""Pure metric functions for the Thai PDPA guard evaluation.

Used by ``scripts/evaluate.py`` (aggregation) and ``scripts/compare.py`` (reading results).
Nothing here touches the filesystem, the GPU or a model.

Conventions
-----------
* ``noul`` questions: probs over ``{"false", "true"}``. Prediction is ``True`` iff
  ``P(true) >= 0.5``. Positive class for precision/recall/F1 is ``True``.
* ``score`` questions (risk_level 0..4): probs over ``"0".."4"``. Prediction is the argmax;
  ``expected`` is ``sum(k * p_k)``.
* ``choice`` questions: probs over the option names. Prediction is the argmax.
* Brier (per question, multi-class form): ``sum_k (p_k - y_k)^2`` where ``y`` is the one-hot
  gold. Range [0, 2]; 0 is perfect. For a noul question this equals ``2 * (P(true) - y)^2``.
  Reported as the mean over questions.
* ECE: **15 equal-width bins** over the confidence ``max_k p_k`` in [0, 1]
  (confidence exactly 1.0 falls into the last bin). ECE = sum over non-empty bins of
  ``n_b / N * |acc_b - conf_b|``. Only the 15-bin version is reported.
* Macro-F1 for ``primary_category`` uses sklearn with ``labels`` = all options and
  ``zero_division=0``. Binary P/R/F1 are computed by hand (``zero_division=0``).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score

IS_SENSITIVE = "is_sensitive"
RISK_LEVEL = "risk_level"
PRIMARY = "primary_category"
HAS_QIDS = (
    "has_name",
    "has_contact",
    "has_address",
    "has_national_id",
    "has_financial",
    "has_credential",
    "has_health_or_special",
    "has_other_identifier",
)
NOUL_KEYS = ("false", "true")
SCORE_KEYS = ("0", "1", "2", "3", "4")
DEFAULT_N_BINS = 15
N_RISK_LEVELS = 5


# --------------------------------------------------------------------------- per-answer helpers
def noul_pred(p_true: float) -> bool:
    """Yes/no prediction: True iff P(true) >= 0.5."""
    return float(p_true) >= 0.5


def argmax_key(probs: Mapping[str, float]) -> str:
    """Key with the highest probability. Ties resolve to the first key in iteration order."""
    if not probs:
        raise ValueError("argmax_key: empty probs")
    return max(probs, key=lambda k: float(probs[k]))


def expected_score(probs: Mapping[str, float]) -> float:
    """Expected score ``sum(int(k) * p_k)`` for score-type probabilities."""
    return float(sum(int(k) * float(p) for k, p in probs.items()))


def label_key(qtype: str, label: Any) -> str:
    """The probs key that corresponds to the gold label."""
    if qtype == "noul":
        return "true" if bool(label) else "false"
    if qtype == "score":
        return str(int(label))
    return str(label)


def is_correct(answer: Mapping[str, Any]) -> bool:
    """Whether the stored prediction equals the gold label, by question type."""
    qtype = answer["type"]
    if qtype == "noul":
        return bool(answer["pred"]) == bool(answer["label"])
    if qtype == "score":
        return int(answer["pred"]) == int(answer["label"])
    return str(answer["pred"]) == str(answer["label"])


# --------------------------------------------------------------------------- scalar metrics
def binary_prf(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict:
    """Precision / recall / F1 for the positive class ``True``. Zero denominators give 0.0."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        t, p = bool(t), bool(p)
        if t and p:
            tp += 1
        elif p and not t:
            fp += 1
        elif t and not p:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / n if n else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support_pos": tp + fn,
    }


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> float | None:
    """Macro-F1 over ``labels`` (sklearn, ``zero_division=0``). None when there is no data."""
    if len(y_true) == 0:
        return None
    return float(
        f1_score(list(y_true), list(y_pred), labels=list(labels), average="macro", zero_division=0)
    )


def per_class_f1(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> dict:
    """Per-label F1 (sklearn, ``zero_division=0``)."""
    if len(y_true) == 0:
        return {}
    vals = f1_score(
        list(y_true), list(y_pred), labels=list(labels), average=None, zero_division=0
    )
    return {str(lab): float(v) for lab, v in zip(labels, vals)}


def risk_stats(gold: Sequence[int], pred_argmax: Sequence[int], expected: Sequence[float]) -> dict:
    """Exact accuracy, +-1 accuracy, and MAE using argmax and using expected score."""
    n = len(gold)
    if n == 0:
        return {"n": 0, "exact_acc": None, "within1_acc": None, "mae_argmax": None, "mae_expected": None}
    g = np.asarray(gold, dtype=float)
    p = np.asarray(pred_argmax, dtype=float)
    e = np.asarray(expected, dtype=float)
    return {
        "n": n,
        "exact_acc": float(np.mean(g == p)),
        "within1_acc": float(np.mean(np.abs(g - p) <= 1)),
        "mae_argmax": float(np.mean(np.abs(g - p))),
        "mae_expected": float(np.mean(np.abs(g - e))),
    }


def latency_stats(values: Iterable[float | None]) -> dict:
    """Mean, p50 and p95 (numpy linear interpolation) of per-record latencies in seconds."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean_s": None, "p50_s": None, "p95_s": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": len(vals),
        "mean_s": float(arr.mean()),
        "p50_s": float(np.percentile(arr, 50)),
        "p95_s": float(np.percentile(arr, 95)),
    }


# --------------------------------------------------------------------------- calibration
def brier_one(probs: Mapping[str, float], label_k: str) -> float:
    """Multi-class Brier score for one question."""
    keys = set(probs) | {label_k}
    return float(
        sum((float(probs.get(k, 0.0)) - (1.0 if k == label_k else 0.0)) ** 2 for k in keys)
    )


def reliability_bins(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = DEFAULT_N_BINS) -> list[dict]:
    """Equal-width bins over [0, 1]. Each bin: lo, hi, n, conf (mean confidence), acc (None if empty)."""
    counts = [0] * n_bins
    conf_sum = [0.0] * n_bins
    corr_sum = [0] * n_bins
    for c, ok in zip(confidences, correct):
        c = min(max(float(c), 0.0), 1.0)
        idx = min(int(c * n_bins), n_bins - 1)
        counts[idx] += 1
        conf_sum[idx] += c
        corr_sum[idx] += 1 if ok else 0
    bins = []
    for i in range(n_bins):
        n = counts[i]
        bins.append(
            {
                "lo": i / n_bins,
                "hi": (i + 1) / n_bins,
                "n": n,
                "conf": conf_sum[i] / n if n else None,
                "acc": corr_sum[i] / n if n else None,
            }
        )
    return bins


def ece_from_bins(bins: Sequence[dict]) -> float | None:
    total = sum(b["n"] for b in bins)
    if total == 0:
        return None
    return float(sum(b["n"] / total * abs(b["acc"] - b["conf"]) for b in bins if b["n"]))


def calibration_stats(answers: Sequence[Mapping[str, Any]], n_bins: int = DEFAULT_N_BINS) -> dict:
    """Brier, ECE and reliability bins over a list of answer dicts (need type, probs, label, pred)."""
    n = len(answers)
    if n == 0:
        return {"n": 0, "brier": None, "ece": None, "bins": reliability_bins([], [], n_bins)}
    briers = [brier_one(a["probs"], label_key(a["type"], a["label"])) for a in answers]
    confs = [max(float(p) for p in a["probs"].values()) for a in answers]
    corr = [is_correct(a) for a in answers]
    bins = reliability_bins(confs, corr, n_bins)
    return {
        "n": n,
        "brier": float(np.mean(briers)),
        "ece": ece_from_bins(bins),
        "bins": bins,
    }


def calibration_for_probs(
    probs_list: Sequence[Mapping[str, float]], label_keys: Sequence[str], n_bins: int = DEFAULT_N_BINS
) -> dict:
    """Lower-level variant of ``calibration_stats`` taking parallel lists (handy in tests)."""
    if not probs_list:
        return {"n": 0, "brier": None, "ece": None, "bins": reliability_bins([], [], n_bins)}
    briers = [brier_one(p, lk) for p, lk in zip(probs_list, label_keys)]
    confs = [max(float(v) for v in p.values()) for p in probs_list]
    corr = [argmax_key(p) == lk for p, lk in zip(probs_list, label_keys)]
    bins = reliability_bins(confs, corr, n_bins)
    return {"n": len(probs_list), "brier": float(np.mean(briers)), "ece": ece_from_bins(bins), "bins": bins}


# --------------------------------------------------------------------------- aggregation
def _answers(rows: Sequence[Mapping[str, Any]], qid: str) -> list[Mapping[str, Any]]:
    out = []
    for r in rows:
        a = r.get("answers", {}).get(qid)
        if a is not None:
            out.append(a)
    return out


def _all_answers(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    out = []
    for r in rows:
        out.extend(r.get("answers", {}).values())
    return out


def headline(rows: Sequence[Mapping[str, Any]], choice_labels: Sequence[str], n_bins: int = DEFAULT_N_BINS) -> dict:
    """Headline metrics over a set of per-question rows (used for the overall and each slice)."""
    out: dict[str, Any] = {"n": len(rows)}

    s = _answers(rows, IS_SENSITIVE)
    if s:
        prf = binary_prf([a["label"] for a in s], [a["pred"] for a in s])
        out.update(
            is_sensitive_precision=prf["precision"],
            is_sensitive_recall=prf["recall"],
            is_sensitive_f1=prf["f1"],
        )
    else:
        out.update(is_sensitive_precision=None, is_sensitive_recall=None, is_sensitive_f1=None)

    f1s = []
    for q in HAS_QIDS:
        a = _answers(rows, q)
        if a:
            f1s.append(binary_prf([x["label"] for x in a], [x["pred"] for x in a])["f1"])
    out["has_macro_f1"] = float(np.mean(f1s)) if f1s else None

    p = _answers(rows, PRIMARY)
    if p:
        labels = list(choice_labels) if choice_labels else sorted({str(x["label"]) for x in p} | {str(x["pred"]) for x in p})
        yt = [str(x["label"]) for x in p]
        yp = [str(x["pred"]) for x in p]
        out["primary_category_accuracy"] = float(np.mean([yt_i == yp_i for yt_i, yp_i in zip(yt, yp)]))
        out["primary_category_macro_f1"] = macro_f1(yt, yp, labels)
    else:
        out.update(primary_category_accuracy=None, primary_category_macro_f1=None)

    r = _answers(rows, RISK_LEVEL)
    rs = risk_stats(
        [int(x["label"]) for x in r],
        [int(x["pred"]) for x in r],
        [float(x["expected"]) for x in r],
    )
    out.update(
        risk_exact_acc=rs["exact_acc"],
        risk_within1_acc=rs["within1_acc"],
        risk_mae_argmax=rs["mae_argmax"],
        risk_mae_expected=rs["mae_expected"],
    )

    cal = calibration_stats(_all_answers(rows), n_bins)
    out["brier"] = cal["brier"]
    out["ece"] = cal["ece"]
    out["latency_mean_s"] = latency_stats([r_.get("latency_s") for r_ in rows])["mean_s"]
    return out


def compute_metrics(
    per_question: Sequence[Mapping[str, Any]],
    choice_labels: Sequence[str] = (),
    n_bins: int = DEFAULT_N_BINS,
) -> dict:
    """Full metrics block written to ``results/eval_<system>.json``.

    ``per_question`` rows: {"id", "genre", "hard_negative", "risk_label", "answers": {qid: answer}, "latency_s"}
    where answer = {"type", "probs", "label", "pred", "correct", ["expected" for score]}.
    """
    rows = list(per_question)
    choice_labels = list(choice_labels)

    qids: list[str] = []
    for r in rows:
        for q in r.get("answers", {}):
            if q not in qids:
                qids.append(q)

    # per-question accuracy
    accuracy_by_qid = {}
    for q in qids:
        a = _answers(rows, q)
        accuracy_by_qid[q] = {
            "type": a[0]["type"],
            "accuracy": float(np.mean([is_correct(x) for x in a])),
            "n": len(a),
        }

    # is_sensitive
    s = _answers(rows, IS_SENSITIVE)
    is_sensitive = binary_prf([x["label"] for x in s], [x["pred"] for x in s]) if s else None

    # has_* nouls
    has = {}
    for q in HAS_QIDS:
        a = _answers(rows, q)
        if a:
            has[q] = binary_prf([x["label"] for x in a], [x["pred"] for x in a])
    has["macro_f1"] = float(np.mean([has[q]["f1"] for q in HAS_QIDS if q in has])) if has else None

    # primary_category
    p = _answers(rows, PRIMARY)
    if p:
        yt = [str(x["label"]) for x in p]
        yp = [str(x["pred"]) for x in p]
        labels = choice_labels or sorted(set(yt) | set(yp))
        primary = {
            "n": len(p),
            "accuracy": float(np.mean([a == b for a, b in zip(yt, yp)])),
            "macro_f1": macro_f1(yt, yp, labels),
            "per_class_f1": per_class_f1(yt, yp, labels),
            "labels": labels,
        }
    else:
        primary = None

    # risk_level (+ confusion matrix rows=gold, cols=argmax)
    r = _answers(rows, RISK_LEVEL)
    if r:
        gold = [int(x["label"]) for x in r]
        pred = [int(x["pred"]) for x in r]
        exp = [float(x["expected"]) for x in r]
        risk = risk_stats(gold, pred, exp)
        conf = [[0] * N_RISK_LEVELS for _ in range(N_RISK_LEVELS)]
        for g, pr in zip(gold, pred):
            if 0 <= g < N_RISK_LEVELS and 0 <= pr < N_RISK_LEVELS:
                conf[g][pr] += 1
        risk["exact_acc"] = risk["exact_acc"]
        risk["confusion"] = conf
        risk["labels"] = list(range(N_RISK_LEVELS))
    else:
        risk = None

    # calibration
    calibration = {
        "n_bins": n_bins,
        "overall": calibration_stats(_all_answers(rows), n_bins),
        "by_type": {},
        "by_qid": {},
    }
    for t in ("noul", "score", "choice"):
        ans_t = [a for a in _all_answers(rows) if a["type"] == t]
        if ans_t:
            calibration["by_type"][t] = calibration_stats(ans_t, n_bins)
    for q in qids:
        calibration["by_qid"][q] = calibration_stats(_answers(rows, q), n_bins)

    # slices
    by_genre: dict[str, Any] = {}
    for g in sorted({str(r.get("genre") or "unknown") for r in rows}):
        sub = [r for r in rows if str(r.get("genre") or "unknown") == g]
        by_genre[g] = headline(sub, choice_labels, n_bins)

    by_gold_risk: dict[str, Any] = {}
    gold_levels = sorted({int(r["risk_label"]) for r in rows if r.get("risk_label") is not None})
    for lvl in gold_levels:
        sub = [r for r in rows if r.get("risk_label") is not None and int(r["risk_label"]) == lvl]
        by_gold_risk[str(lvl)] = headline(sub, choice_labels, n_bins)

    by_hard_negative: dict[str, Any] = {}
    for flag in (True, False):
        sub = [r for r in rows if bool(r.get("hard_negative")) == flag]
        if sub:
            by_hard_negative["true" if flag else "false"] = headline(sub, choice_labels, n_bins)

    return {
        "n_records": len(rows),
        "n_answers": len(_all_answers(rows)),
        "choice_labels": choice_labels,
        "accuracy_by_qid": accuracy_by_qid,
        "is_sensitive": is_sensitive,
        "has": has,
        "primary_category": primary,
        "risk_level": risk,
        "calibration": calibration,
        "latency": latency_stats([r.get("latency_s") for r in rows]),
        "headline": headline(rows, choice_labels, n_bins),
        "by_genre": by_genre,
        "by_gold_risk": by_gold_risk,
        "by_hard_negative": by_hard_negative,
    }


# --------------------------------------------------------------------------- JSON sanitising
def jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to Python, NaN/inf to None, tuples to lists."""
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj
