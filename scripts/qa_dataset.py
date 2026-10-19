"""QA for the generated dataset. Exits non-zero on any failure; writes data/processed/QA_REPORT.md.
Usage: python scripts/qa_dataset.py [--dir data/processed]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdpa_guard import fakers  # noqa: E402
from pdpa_guard.taxonomy import (CATEGORIES, PII_TYPES, QUESTION_IDS, labels_from_spans)  # noqa: E402

TARGET = {"train": 2700, "dev": 450, "test": 735}
MIX = {"positive": 0.55, "hard_negative": 0.25, "clean_negative": 0.20}
fails: list[str] = []
lines: list[str] = []


def fail(msg):
    if len(fails) < 200:
        fails.append(msg)


def out(s=""):
    lines.append(s)
    print(s)


def stext(r):
    return r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)


def shingles(t, k=5):
    t = re.sub(r"\s+", " ", t.lower())
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def minhash(sh, n=64):
    return tuple(min(int.from_bytes(hashlib.md5(f"{i}|{s}".encode()).digest()[:6], "big") for s in sh)
                 for i in range(n))


def near_dups(items, thresh=0.8):
    """items: list of (key, set). Returns list of (key_i, key_j, jaccard) via LSH banding + exact check."""
    sigs = [minhash(s) for _, s in items]
    bands, rows = 32, 2
    buckets = defaultdict(list)
    for idx, sg in enumerate(sigs):
        for b in range(bands):
            buckets[(b, sg[b * rows:(b + 1) * rows])].append(idx)
    seen, res = set(), []
    for lst in buckets.values():
        if len(lst) < 2 or len(lst) > 400:
            continue
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                p = (lst[a], lst[b])
                if p in seen:
                    continue
                seen.add(p)
                A, B = items[p[0]][1], items[p[1]][1]
                j = len(A & B) / max(1, len(A | B))
                if j >= thresh:
                    res.append((items[p[0]][0], items[p[1]][0], j))
    return res


def check_record(r, split):
    rid = r["id"]
    m = r["meta"]
    text = stext(r)
    spans = m["pii_spans"]
    q = r["questions"]
    if list(q) != QUESTION_IDS:
        fail(f"{rid}: question ids {list(q)}")
        return
    exp = labels_from_spans(spans)
    for qid in QUESTION_IDS:
        if q[qid]["label"] != exp[qid]:
            fail(f"{rid}: label {qid} stored {q[qid]['label']} != recomputed {exp[qid]}")
    if q["risk_level"]["type"] != "score" or len(q["risk_level"]["criteria"]) != 5:
        fail(f"{rid}: risk question malformed")
    if set(q["primary_category"]["criteria"]) != set(CATEGORIES) or q["primary_category"]["type"] != "choice":
        fail(f"{rid}: primary_category malformed")
    for qid in QUESTION_IDS:
        if not q[qid].get("instructions"):
            fail(f"{rid}: empty instructions {qid}")
        if qid not in ("risk_level", "primary_category") and q[qid]["type"] != "noul":
            fail(f"{rid}: {qid} should be noul")
    covered = []
    for s in spans:
        if s["type"] not in PII_TYPES:
            fail(f"{rid}: bad span type {s['type']}")
        if s["value"] not in text:
            fail(f"{rid}: span value not in text {s['value']!r}")
        digits = fakers.normalize_digits(s["value"])
        if s["type"] == "national_id":
            if not fakers.is_valid_thai_id(s["value"]):
                fail(f"{rid}: invalid national_id span {s['value']}")
        if s["type"] == "card_number":
            d = re.sub(r"\D", "", digits)[:16]
            if len(d) != 16 or not fakers.luhn_ok(d):
                fail(f"{rid}: card span fails Luhn {s['value']}")
            if not d.startswith(("4", "51", "52", "53", "54", "55", "35", "22", "23", "24", "25", "26", "27")):
                fail(f"{rid}: card BIN {d[:4]}")
        if s["type"] == "passport" and not re.fullmatch(r"[A-Z]{1,2}\d{7}", digits.strip()):
            fail(f"{rid}: passport format {s['value']}")
        covered.append(digits)
    # look-alike scan: any valid-looking id/card in the text must be a labelled span
    ndig = fakers.normalize_digits(text)
    covered_digits = " ".join(re.sub(r"\D", "", c) for c in covered)
    for mm in re.finditer(r"(?<!\d)\d(?:[ -]?\d){12}(?!\d)", ndig):
        d = re.sub(r"\D", "", mm.group())
        if len(d) == 13 and fakers.is_valid_thai_id(d) and d not in covered_digits:
            fail(f"{rid}: unlabelled valid-checksum 13-digit number {mm.group()}")
    for mm in re.finditer(r"(?<!\d)\d(?:[ -]?\d){15}(?!\d)", ndig):
        d = re.sub(r"\D", "", mm.group())
        if len(d) == 16 and fakers.luhn_ok(d) and d[:1] in "45" and d not in covered_digits:
            fail(f"{rid}: unlabelled Luhn-valid 16-digit number {mm.group()}")
    kind, risk = m["kind"], exp["risk_level"]
    if kind == "positive" and risk < 1:
        fail(f"{rid}: positive with risk 0")
    if kind == "hard_negative" and risk > 1:
        fail(f"{rid}: hard negative risk {risk}")
    if kind == "clean_negative" and spans:
        fail(f"{rid}: clean negative has spans")
    if m["hard_negative"] != (kind == "hard_negative"):
        fail(f"{rid}: hard_negative flag")
    if m["split"] != split:
        fail(f"{rid}: meta.split {m['split']} != {split}")
    tid = m["template_id"]
    if split == "test" and not tid.startswith(("T-B-", "N-B-")):
        fail(f"{rid}: non-B template {tid} in test")
    if split != "test" and not tid.startswith(("T-A-", "N-A-")):
        fail(f"{rid}: non-A template {tid} in {split}")
    if tid.startswith("N-") and not 250 <= len(text) <= 2300:
        fail(f"{rid}: long-form length {len(text)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--tolerance", type=float, default=0.05)
    a = ap.parse_args()
    d = Path(a.dir)
    data = {}
    for sp in ("train", "dev", "test"):
        data[sp] = [json.loads(l) for l in (d / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        kev = [json.loads(l) for l in (d / "kev" / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(kev) != len(data[sp]) or any("meta" in k for k in kev):
            fail(f"kev/{sp}.jsonl mismatched or contains meta")
        elif any(k["questions"] != r["questions"] or k["state"] != r["state"] for k, r in zip(kev, data[sp])):
            fail(f"kev/{sp}.jsonl differs from full file")
    out("# QA REPORT\n")
    ids = Counter(r["id"] for sp in data for r in data[sp])
    for i, c in ids.items():
        if c > 1:
            fail(f"duplicate id {i}")
    for sp, rs in data.items():
        out(f"## {sp}: {len(rs)} records (target {TARGET[sp]})")
        if abs(len(rs) - TARGET[sp]) > 0.06 * TARGET[sp]:
            fail(f"{sp}: size {len(rs)} vs target {TARGET[sp]}")
        for r in rs:
            check_record(r, sp)
        kinds = Counter(r["meta"]["kind"] for r in rs)
        out("kind mix: " + ", ".join(f"{k}={kinds[k]} ({kinds[k] / len(rs):.1%})" for k in MIX))
        for k, f in MIX.items():
            if abs(kinds[k] / len(rs) - f) > a.tolerance:
                fail(f"{sp}: kind {k} share {kinds[k] / len(rs):.2f} vs {f}")
        out("\nclass balance:")
        for qid in QUESTION_IDS:
            c = Counter(str(r["questions"][qid]["label"]) for r in rs)
            out(f"- {qid}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
        rc = Counter(r["questions"]["risk_level"]["label"] for r in rs)
        out("\nrisk histogram: " + ", ".join(f"{k}:{rc.get(k, 0)}" for k in range(5)))
        if any(rc.get(k, 0) < 0.03 * len(rs) for k in range(5)):
            fail(f"{sp}: a risk level has <3% of records: {dict(rc)}")
        gc = Counter(r["meta"]["genre"] for r in rs)
        out("genre histogram: " + ", ".join(f"{k}={v}" for k, v in sorted(gc.items())))
        if len(gc) < 12:
            fail(f"{sp}: only {len(gc)} genres")
        tc = Counter(r["meta"]["template_id"] for r in rs if not r["meta"]["template_id"].startswith("N-"))
        nlong = sum(1 for r in rs if r["meta"]["template_id"].startswith("N-"))
        out(f"templates used: {len(tc)}, max instances/template: {max(tc.values())}, free-form long records: {nlong}")
        if max(tc.values()) > 20:
            fail(f"{sp}: a template instantiated {max(tc.values())} times")
        L = [len(stext(r)) for r in rs]
        out(f"length chars: min={min(L)} median={statistics.median(L)} mean={statistics.mean(L):.0f} max={max(L)}")
        cov = Counter()
        for r in rs:
            for s_ in {x["type"] for x in r["meta"]["pii_spans"]}:
                cov[s_] += 1
        out("records containing each span type: " + ", ".join(f"{k}={v}" for k, v in sorted(cov.items())))
        langs = Counter(r["meta"]["lang"] for r in rs)
        out(f"instruction language: {dict(langs)}")
        tp = Counter(s["type"] for r in rs for s in r["meta"]["pii_spans"])
        out("span types: " + ", ".join(f"{k}={v}" for k, v in sorted(tp.items())) + "\n")
    # leakage
    out("## leakage")
    tr = {r["meta"]["template_id"] for r in data["train"]}
    dv = {r["meta"]["template_id"] for r in data["dev"]}
    te = {r["meta"]["template_id"] for r in data["test"]}
    out(f"template overlap train/dev={len(tr & dv)}, train/test={len(tr & te)}, dev/test={len(dv & te)}")
    if tr & dv or tr & te or dv & te:
        fail("template ids shared between splits")
    if any("T-B" in t for t in tr | dv):
        fail("T-B template in train/dev")
    # near duplicates
    out("\n## near-duplicates (char 5-gram Jaccard >= 0.8)")
    allitems = []
    for sp, rs in data.items():
        for r in rs:
            allitems.append(((sp, r["id"], r["meta"]["template_id"]), shingles(stext(r))))
    pairs = near_dups(allitems)
    within = Counter()
    cross = []
    for ka, kb, j in pairs:
        if ka[0] == kb[0]:
            within[ka[0]] += 1
        else:
            cross.append((ka, kb, j))
    for sp, rs in data.items():
        out(f"- {sp}: {within[sp]} near-dup pairs within split ({within[sp] / max(1, len(rs)):.2%} per record)")
    out(f"- cross-split near-dup pairs: {len(cross)}")
    for ka, kb, j in cross[:10]:
        out(f"  - {ka[1]} ({ka[2]}) ~ {kb[1]} ({kb[2]}) J={j:.2f}")
    if cross:
        fail(f"{len(cross)} cross-split near-duplicate pairs")
    byt = Counter()
    for ka, kb, j in pairs:
        if ka[0] == kb[0] and ka[2] == kb[2]:
            byt[ka[2]] += 1
    wide = [t for t, c in byt.items() if c > 40]
    out(f"- templates with >40 intra-template near-dup pairs (long template, short slots): {len(wide)}")
    out("\n## result")
    if fails:
        out(f"FAILED ({len(fails)} problems, first 30):")
        for f in fails[:30]:
            out(f"- {f}")
    else:
        out("ALL CHECKS PASSED")
    (d / "QA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
