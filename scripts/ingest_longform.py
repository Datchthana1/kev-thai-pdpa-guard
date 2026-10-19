"""Validate writer outputs for free-form long records and build data/longform/N_{A,B}.jsonl.

Spans/labels come from the value pack (fakers), never from the writer. A record is dropped when
  - a pack value is not verbatim in the text, or length is outside 250..2200 chars,
  - the text contains extra PII-like strings (phones/emails/IDs/cards/IPs/titled names) not in the pack,
  - sensitive health/religion/crime keywords appear although the pack has no health/special/stat value.
Usage: python scripts/ingest_longform.py
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdpa_guard import fakers  # noqa: E402

LF = ROOT / "data" / "longform"
PHONE = re.compile(r"(?<!\d)(?:\+66|0066|66)?[\s-]?\(?0?\)?[689]\d[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
TITLED = re.compile(r"(?<![฀-๿])(?:คุณ|นาย|นางสาว|นาง|น\.ส\.|ด\.ช\.|ด\.ญ\.|ดร\.|นพ\.|พญ\.|Mr\.|Mrs\.|Ms\.|Dr\.)\s?([A-Za-z฀-๿]{2,15})")
EXEMPT_PREFIX = ("ภาพ", "ลูกค้า", "ครู", "ทะเบียน", "พยาบาล", "ปู่", "ย่า", "เรียบร้อย", "ระบบ", "พ่อ", "แม่", "หมอ", "ผู้", "ที่", "ใด", "นี้", "สมาชิก", "เจ้า", "ทุก", "ได้", "ใน", "และ",
                 "ของ", "มา", "ไป", "จะ", "ก็", "ท่าน", "ตัว", "หนึ่ง", "บริษัท", "ธนาคาร", "ล่วง", "เป็น", "ช่วย", "โปรด", "กรุณา",
                 "ว่า", "ต้อง", "ยัง", "อยู่", "รับ", "ขอ", "แจ้ง", "เรียน", "ตรวจ", "ผล", "สะดวก", "พี่", "น้อง", "ลุง", "ป้า", "ยาย",
                 "สอบ", "ซื้อ", "ลง", "ส่ง", "ติด", "จัด", "ทำ", "เลย", "ค่ะ", "ครับ", "นะ")
SENS = re.compile(r"เบาหวาน|มะเร็ง|ซึมเศร้า|HIV|เอชไอวี|ตั้งครรภ์|ศาสนา|อิสลาม|มุสลิม|คริสต์|พุทธ|คดี|จำคุก|พรรค|LGBT|เกย์|ความดัน|โรค|ผ่าตัด|ยาเสพติด|ลายนิ้วมือ|พิการ")


def digits(s):
    return re.sub(r"\D", "", fakers.normalize_digits(s))


def text_of(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def validate(pack, state):
    txt = text_of(state)
    if not 250 <= len(txt) <= 2200:
        return f"length {len(txt)}"
    for v in pack["values"]:
        if v["value"] not in txt:
            return f"missing value {v['key']}"
    allvals = " ".join(v["value"] for v in pack["values"])
    valdigits = {digits(v["value"]) for v in pack["values"]}
    for m in PHONE.finditer(fakers.normalize_digits(txt)):
        d = digits(m.group())
        if not any(d and (d in vd or vd.endswith(d) or d.endswith(vd[-9:] if len(vd) >= 9 else "#")) for vd in valdigits if vd):
            return f"extra phone {m.group()}"
    for m in EMAIL.finditer(txt):
        if m.group() not in allvals:
            return f"extra email {m.group()}"
    for m in IPV4.finditer(txt):
        if m.group() not in allvals and not re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{1,2}\.\d{1,2}", m.group()) is None:
            return f"extra ip {m.group()}"
    ntxt = fakers.normalize_digits(txt)
    for m in re.finditer(r"(?<!\d)\d(?:[ -]?\d){12,15}(?!\d)", ntxt):
        d = re.sub(r"\D", "", m.group())
        if d not in valdigits and not any(d in vd for vd in valdigits):
            return f"extra long number {m.group()}"
    for m in TITLED.finditer(txt):
        tok = m.group(1)
        if tok.startswith(EXEMPT_PREFIX):
            continue
        if tok[:3] not in allvals and tok[:2] not in allvals:
            return f"extra titled name {m.group()}"
    keys = {v["key"] for v in pack["values"]}
    if not (keys & {"health", "special", "health_stat", "special_stat"}) and SENS.search(txt):
        return "sensitive keyword without pack value"
    return None


def main():
    rng = random.Random(3)
    for side in "AB":
        packs = {}
        for f in sorted((LF / f"packs_{side}").glob("pack_*.json")):
            for p in json.loads(f.read_text(encoding="utf-8")):
                packs[p["pack_id"]] = p
        recs, drops, seen = [], Counter(), set()
        for f in sorted((LF / f"out_{side}").glob("out_*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    o = json.loads(line)
                    pack = packs[o["pack_id"]]
                    state = o["state"]
                except Exception:
                    drops["bad json/unknown pack"] += 1
                    continue
                if o["pack_id"] in seen:
                    drops["duplicate pack"] += 1
                    continue
                seen.add(o["pack_id"])
                if pack["kind"] != "positive" or True:
                    pass
                err = validate(pack, state)
                if err:
                    drops[err.split(" ")[0] + " " + (err.split(" ")[1] if " " in err else "")] += 1
                    continue
                recs.append({"genre": pack["genre"], "kind": pack["kind"], "state": state, "pii_spans": pack["spans"],
                             "hard_negative": pack["kind"] == "hard_negative", "pack_id": pack["pack_id"]})
        rng.shuffle(recs)
        for i, r in enumerate(recs, 1):
            r["template_id"] = f"N-{side}-{i:03d}"
        (LF / f"N_{side}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
        print(side, "packs", len(packs), "written", len(seen), "kept", len(recs), "kinds", dict(Counter(r["kind"] for r in recs)))
        print("  drops:", dict(drops))


if __name__ == "__main__":
    main()
