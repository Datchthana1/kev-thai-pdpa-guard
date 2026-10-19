#!/usr/bin/env python
"""Evaluate one system on the PDPA test split with identical questions for every system.

Examples (repo root, venv python):
  .venv/Scripts/python.exe scripts/evaluate.py --system kev-base --test data/processed/test.jsonl --device cuda --load_4bit 1
  .venv/Scripts/python.exe scripts/evaluate.py --system kev-pdpa --run runs/<dir> --test data/processed/test.jsonl --device cuda --load_4bit 1
  .venv/Scripts/python.exe scripts/evaluate.py --system openthai --limit 50 --device cpu

Every system receives the same stripped questions: {qid: {"type", "instructions", "criteria"?}} with
labels and meta removed. Output: results/eval_<system>.json (written once at the end).
While running, each finished record is appended to results/eval_<system>.partial.jsonl, so an
interrupted run resumes from that file (use --fresh to ignore it).

Set EVAL_FAKE=1 to use the random fake system from tests/fake_system.py instead of kev_qlora.systems.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kev_qlora.metrics_pdpa import (  # noqa: E402
    DEFAULT_N_BINS,
    NOUL_KEYS,
    PRIMARY,
    RISK_LEVEL,
    SCORE_KEYS,
    argmax_key,
    compute_metrics,
    expected_score,
    is_correct,
    jsonable,
    noul_pred,
)

SYSTEMS = ("kev-base", "kev-pdpa", "openthai")
QTYPES = ("noul", "score", "choice")


# --------------------------------------------------------------------------- input
def load_records(path: Path, limit: int | None) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            _validate_record(rec, f"{path}:{lineno}")
            records.append(rec)
            if limit and len(records) >= limit:
                break
    if not records:
        raise SystemExit(f"no records read from {path}")
    return records


def _validate_record(rec: dict, where: str) -> None:
    for key in ("id", "state", "questions"):
        if key not in rec:
            raise SystemExit(f"{where}: missing '{key}'")
    for qid, q in rec["questions"].items():
        if q.get("type") not in QTYPES:
            raise SystemExit(f"{where}: question {qid} has bad type {q.get('type')!r}")
        if "instructions" not in q:
            raise SystemExit(f"{where}: question {qid} has no instructions")
        if "label" not in q:
            raise SystemExit(f"{where}: question {qid} has no label (test split must be labelled)")


def public_questions(questions: dict) -> dict:
    """Strip labels and meta: keep only type, instructions and criteria (exactly as in the record)."""
    out = {}
    for qid, q in questions.items():
        d = {"type": q["type"], "instructions": q["instructions"]}
        if "criteria" in q:
            d["criteria"] = copy.deepcopy(q["criteria"])
        out[qid] = d
    return out


def choice_labels_from(records: list[dict]) -> list[str]:
    for rec in records:
        q = rec["questions"].get(PRIMARY)
        if q and isinstance(q.get("criteria"), dict):
            return list(q["criteria"].keys())
    return []


# --------------------------------------------------------------------------- output rows
def expected_keys(q: dict) -> list[str]:
    if q["type"] == "noul":
        return list(NOUL_KEYS)
    if q["type"] == "score":
        if isinstance(q.get("criteria"), list):
            return [str(i) for i in range(len(q["criteria"]))]
        return list(SCORE_KEYS)
    if not isinstance(q.get("criteria"), dict):
        raise ValueError("choice question needs a criteria dict")
    return list(q["criteria"].keys())


def normalize_answer(qid: str, q: dict, raw: dict | None, rec_id: str) -> dict:
    """Validate one system answer against the question and return the stored answer dict."""
    if raw is None or "probs" not in raw:
        raise ValueError(f"{rec_id}/{qid}: system returned no probs")
    if raw.get("type") not in (None, q["type"]):
        raise ValueError(f"{rec_id}/{qid}: answer type {raw.get('type')!r} != question type {q['type']!r}")
    keys = expected_keys(q)
    probs_in = raw["probs"]
    missing = [k for k in keys if k not in probs_in]
    extra = [k for k in probs_in if k not in keys]
    if missing or extra:
        raise ValueError(f"{rec_id}/{qid}: probs keys mismatch, missing={missing} extra={extra}")
    vals = [float(probs_in[k]) for k in keys]
    if any((v < 0) or (v != v) or (v == float("inf")) for v in vals):
        raise ValueError(f"{rec_id}/{qid}: negative or non-finite probability")
    total = sum(vals)
    if total <= 0:
        raise ValueError(f"{rec_id}/{qid}: probabilities sum to zero")
    probs = {k: v / total for k, v in zip(keys, vals)}  # renormalise (no-op when already sums to 1)

    gold = q["label"]
    ans: dict = {"type": q["type"], "probs": probs}
    if q["type"] == "noul":
        ans["label"] = bool(gold)
        ans["pred"] = noul_pred(probs["true"])
    elif q["type"] == "score":
        ans["label"] = int(gold)
        ans["pred"] = int(argmax_key(probs))
        ans["expected"] = expected_score(probs)
    else:
        ans["label"] = str(gold)
        ans["pred"] = argmax_key(probs)
    ans["correct"] = is_correct(ans)
    return ans


def build_row(rec: dict, raw_answers: dict, latency_s: float) -> dict:
    answers = {}
    for qid, q in rec["questions"].items():
        answers[qid] = normalize_answer(qid, q, raw_answers.get(qid), rec["id"])
    meta = rec.get("meta") or {}
    risk_q = rec["questions"].get(RISK_LEVEL)
    return {
        "id": rec["id"],
        "genre": meta.get("genre"),
        "hard_negative": bool(meta.get("hard_negative", False)),
        "risk_label": int(risk_q["label"]) if risk_q else None,
        "answers": answers,
        "latency_s": float(latency_s),
    }


# --------------------------------------------------------------------------- system loading
def load_system(name: str, run: str | None, device: str, load_4bit: bool):
    """Lazy import so tests can monkeypatch; EVAL_FAKE=1 swaps in the fake system."""
    if os.environ.get("EVAL_FAKE") == "1":
        tests_dir = REPO / "tests"
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        from fake_system import get_system as fake_get_system

        return fake_get_system(name, run=run, device=device, load_4bit=load_4bit)
    from kev_qlora.systems import get_system

    return get_system(name, run=run, device=device, load_4bit=load_4bit)


# --------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True, choices=SYSTEMS)
    p.add_argument("--test", default=str(REPO / "data" / "processed" / "test.jsonl"))
    p.add_argument("--run", default=None, help="run dir (required for kev-pdpa)")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N records")
    p.add_argument("--out", default=None, help="default: results/eval_<system>.json")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--load_4bit", type=int, default=0, choices=[0, 1])
    p.add_argument("--fresh", action="store_true", help="ignore an existing .partial.jsonl")
    p.add_argument("--resume", type=int, default=1, choices=[0, 1],
                   help="1 (default): skip records already in results/eval_<system>.partial.jsonl; 0 = same as --fresh")
    args = p.parse_args(argv)
    if args.system == "kev-pdpa" and not args.run:
        p.error("--run is required for --system kev-pdpa")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.resume:
        args.fresh = True
    test_path = Path(args.test)
    if not test_path.is_absolute():
        test_path = (REPO / test_path).resolve() if not test_path.exists() else test_path.resolve()
    out_path = Path(args.out) if args.out else REPO / "results" / f"eval_{args.system}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_name(out_path.stem + ".partial.jsonl")

    records = load_records(test_path, args.limit)
    n = len(records)
    choice_labels = choice_labels_from(records)
    fake = os.environ.get("EVAL_FAKE") == "1"

    done: dict[str, dict] = {}
    if partial_path.exists() and not args.fresh:
        with open(partial_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done[row["id"]] = row
    resumed_ids = {r["id"] for r in records if r["id"] in done}

    system = None
    if len(resumed_ids) < n:
        system = load_system(args.system, args.run, args.device, bool(args.load_4bit))

    mode = "a" if (partial_path.exists() and not args.fresh) else "w"
    with open(partial_path, mode, encoding="utf-8") as pf:
        for i, rec in enumerate(records, 1):
            if rec["id"] in done:
                row = done[rec["id"]]
                print(f"rec={i}/{n} id={rec['id']} latency={row['latency_s']:.3f}s resumed", flush=True)
                continue
            questions = public_questions(rec["questions"])
            raw, latency = system.predict(rec["state"], questions)
            row = build_row(rec, raw, latency)
            row["input_sha"] = hashlib.sha256(json.dumps([rec["state"], questions], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
            pf.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
            pf.flush()
            os.fsync(pf.fileno())
            done[rec["id"]] = row
            print(f"rec={i}/{n} id={rec['id']} latency={latency:.3f}s", flush=True)

    per_question = [done[r["id"]] for r in records]
    metrics = compute_metrics(per_question, choice_labels, n_bins=DEFAULT_N_BINS)
    result = {
        "system": args.system,
        "n_records": n,
        "config": {
            "test": str(test_path),
            "limit": args.limit,
            "run": args.run,
            "device": args.device,
            "load_4bit": bool(args.load_4bit),
            "fake": fake,
            "resumed_records": len(resumed_ids),
            "calibration_bins": DEFAULT_N_BINS,
            "noul_threshold": 0.5,
            "choice_labels": choice_labels,
        },
        "per_question": per_question,
        "metrics": metrics,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jsonable(result), f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)

    h = metrics["headline"]
    print(
        f"wrote {out_path} | is_sensitive F1={_fmt(h.get('is_sensitive_f1'))} "
        f"risk exact={_fmt(h.get('risk_exact_acc'))} primary acc={_fmt(h.get('primary_category_accuracy'))} "
        f"has macro-F1={_fmt(h.get('has_macro_f1'))}",
        flush=True,
    )
    return 0


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    sys.exit(main())
