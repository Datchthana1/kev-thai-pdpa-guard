"""Dataset generator.  Usage:
    python -m pdpa_guard.generate [--seed 42] [--train 2400 --dev 400 --test 600] [--out data/processed]
(run from the project root with PYTHONPATH=src, or just `python src/pdpa_guard/generate.py`).

Templates data/templates/train_A.jsonl (T-A-###) feed train + dev (a stratified ~12% of A templates is held out
for dev only); data/templates/test_B.jsonl (T-B-###) feeds test only.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pdpa_guard import fakers  # noqa: E402
from pdpa_guard.taxonomy import build_questions, compute_risk  # noqa: E402

SLOT_RE = re.compile(r"\{([a-z_]+?)(\d*)\}")
MIX = {"positive": 0.55, "hard_negative": 0.25, "clean_negative": 0.20}
MAX_PER_TEMPLATE = 15


def load_templates(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


_TITLE_DUP = re.compile(r"(คุณ|นาย|นางสาว|นาง|น\.ส\.|ด\.ช\.|ด\.ญ\.|ดร\.)\1")
_WORD_DUP = re.compile(r"(?<!\S)(บาท|น\.|ร้อยละ|เป็น|ว่า)\s?\1")


def _dedupe_words(s):
    """Remove accidental doubled tokens caused by template text + slot text ("บาท บาท", "คุณคุณ")."""
    s = _TITLE_DUP.sub(r"\1", s)
    return _WORD_DUP.sub(r"\1", s)


def _fill_str(s, rng, cache, spans):
    def sub(m):
        base, suf = m.group(1), m.group(2)
        key = base + suf
        if key not in cache:
            text, sp = fakers.gen_slot(base, rng)
            cache[key] = text
            spans.extend(sp)
        return cache[key]
    return _dedupe_words(SLOT_RE.sub(sub, s))


def _fill_obj(o, rng, cache, spans):
    if isinstance(o, str):
        return _fill_str(o, rng, cache, spans)
    if isinstance(o, dict):
        return {k: _fill_obj(v, rng, cache, spans) for k, v in o.items()}
    if isinstance(o, list):
        return [_fill_obj(v, rng, cache, spans) for v in o]
    return o


def state_text(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def instantiate(tpl, rng):
    cache, spans = {}, []
    if "text" in tpl:
        state = _fill_str(tpl["text"], rng, cache, spans)
    else:
        state = _fill_obj(copy.deepcopy(tpl["json"]), rng, cache, spans)
    txt = state_text(state)
    uniq, seen = [], set()
    for s in spans:
        if s["value"] not in txt:
            raise ValueError(f"span value not in text: {s} / {tpl['template_id']}")
        k = (s["type"], s["value"])
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return state, uniq


def norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def split_dev_templates(a_templates, rng, frac=0.12):
    groups = defaultdict(list)
    for t in a_templates:
        groups[(t["genre"], t["kind"])].append(t)
    dev = set()
    for k, lst in sorted(groups.items()):
        lst = sorted(lst, key=lambda t: t["template_id"])
        rng.shuffle(lst)
        n = max(1, round(len(lst) * frac)) if len(lst) >= 3 else 0
        dev.update(t["template_id"] for t in lst[:n])
    return dev


def make_split(templates, n_total, split, rng, prefix, counter_start=0, seen_texts=None, extra=()):
    seen_texts = seen_texts if seen_texts is not None else set()
    by_kind = defaultdict(list)
    for t in templates:
        by_kind[t["kind"]].append(t)
    targets = {k: round(n_total * f) for k, f in MIX.items()}
    targets["positive"] += n_total - sum(targets.values())
    records = []
    for kind, tgt in targets.items():
        tpls = by_kind[kind]
        if not tpls:
            raise SystemExit(f"no templates of kind {kind} for split {split}")
        # balance across genres first, then templates
        counts = defaultdict(int)
        order = list(tpls)
        made = 0
        cap = MAX_PER_TEMPLATE
        stall = 0
        while made < tgt and stall < 3:
            progressed = False
            rng.shuffle(order)
            for t in sorted(order, key=lambda t: counts[t["template_id"]]):
                if made >= tgt:
                    break
                if counts[t["template_id"]] >= cap:
                    continue
                for _ in range(12):  # retry for a non-duplicate text
                    state, spans = instantiate(t, rng)
                    key = norm(state_text(state))
                    if key not in seen_texts:
                        break
                else:
                    counts[t["template_id"]] = cap  # template exhausted (no/low variety)
                    continue
                seen_texts.add(key)
                counts[t["template_id"]] += 1
                made += 1
                progressed = True
                records.append((t, state, spans))
            if not progressed:
                cap += 5  # relax cap only if nothing could be made
                stall += 1 if cap > 40 else 0
        if made < tgt:
            print(f"WARN {split}/{kind}: only {made}/{tgt}", file=sys.stderr)
    for e in extra:  # free-form long records (fixed text, spans from value packs)
        records.append(({"genre": e["genre"], "kind": e["kind"], "template_id": e["template_id"]}, e["state"], e["pii_spans"]))
    rng.shuffle(records)
    out, outk = [], []
    for i, (t, state, spans) in enumerate(records, counter_start + 1):
        qrng = random.Random(rng.random())
        lang = "en" if qrng.random() < 0.15 else "th"
        questions = build_questions(spans, qrng, lang)
        rid = f"{prefix}-{i:06d}"
        meta = {"genre": t["genre"], "template_id": t["template_id"], "split": split, "kind": t["kind"],
                "lang": lang, "pii_spans": spans, "hard_negative": t["kind"] == "hard_negative",
                "risk": compute_risk(spans)}
        out.append({"id": rid, "state": state, "questions": questions, "meta": meta})
        outk.append({"id": rid, "state": state, "questions": questions})
    return out, outk


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=int, default=2400)
    ap.add_argument("--dev", type=int, default=400)
    ap.add_argument("--test", type=int, default=560)
    ap.add_argument("--longform", default=str(ROOT / "data" / "longform"))
    ap.add_argument("--dev_long", type=int, default=50)
    ap.add_argument("--templates", default=str(ROOT / "data" / "templates"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    tdir = Path(a.templates)
    A = load_templates(tdir / "train_A.jsonl")
    B = load_templates(tdir / "test_B.jsonl")
    assert all(t["template_id"].startswith("T-A-") for t in A) and all(t["template_id"].startswith("T-B-") for t in B)
    need = [{"password", "otp", "pin", "api_key"}, {"health"}, {"special"}, {"national_id", "passport"},
            {"bank_account", "card", "card_full"}, {"plate", "employee_id", "student_id", "ip", "line_id"},
            {"phone", "email"}, {"address"}]
    for k in range(500):  # pick a dev template set that covers every PII family with >=3 templates each
        dev_ids = split_dev_templates(A, random.Random(a.seed + 1 + k))
        cnt = [0] * len(need)
        for t in A:
            if t["template_id"] in dev_ids and t["kind"] == "positive":
                sl = {m.group(1) for m in SLOT_RE.finditer(json.dumps(t, ensure_ascii=False))}
                for i, n_ in enumerate(need):
                    cnt[i] += bool(sl & n_)
        if all(c >= 3 for c in cnt):
            break
    train_t = [t for t in A if t["template_id"] not in dev_ids]
    dev_t = [t for t in A if t["template_id"] in dev_ids]
    lf = Path(a.longform)
    NA = load_templates(lf / "N_A.jsonl") if (lf / "N_A.jsonl").exists() else []
    NB = load_templates(lf / "N_B.jsonl") if (lf / "N_B.jsonl").exists() else []
    lrng = random.Random(a.seed + 99)
    lrng.shuffle(NA)
    dev_long, train_long = NA[:a.dev_long], NA[a.dev_long:]
    extras = {"train": train_long, "dev": dev_long, "test": NB}
    seen = set()  # global dedupe of normalised text across all splits
    out = Path(a.out)
    for name, tpls, n, prefix in (("train", train_t, a.train, "tr"), ("dev", dev_t, a.dev, "dv"),
                                  ("test", B, a.test, "te")):
        full, kev = make_split(tpls, n, name, random.Random(rng.random()), prefix, seen_texts=seen, extra=extras[name])
        write_jsonl(out / f"{name}.jsonl", full)
        write_jsonl(out / "kev" / f"{name}.jsonl", kev)
        print(f"{name}: {len(full)} records from {len(tpls)} templates")


if __name__ == "__main__":
    main()
