"""Writes tests/fixtures/tiny_train.jsonl (6 hand-made Kev-format records, no meta) for smoke tests."""
import json
from pathlib import Path

HAS = {"has_name": "มีชื่อบุคคลหรือไม่", "has_contact": "มีเบอร์โทรหรืออีเมลหรือไม่", "has_address": "มีที่อยู่หรือไม่",
       "has_national_id": "มีเลขบัตรประชาชนหรือหนังสือเดินทางหรือไม่", "has_financial": "มีเลขบัญชีหรือบัตรหรือไม่",
       "has_credential": "มีรหัสผ่านหรือ OTP หรือไม่", "has_health_or_special": "มีข้อมูลสุขภาพหรืออ่อนไหวหรือไม่",
       "has_other_identifier": "มีรหัสระบุตัวตนอื่นหรือไม่"}
CATS = ["none", "name", "contact", "address", "other_identifier", "national_id", "financial", "credential", "health", "special_category"]
RISK = ["ไม่มีข้อมูลส่วนบุคคล", "ความเสี่ยงต่ำ", "ความเสี่ยงปานกลาง", "ความเสี่ยงสูง", "ความเสี่ยงวิกฤต"]

# (text, has-flags, primary, risk)
ROWS = [
    ("สวัสดีครับ พรุ่งนี้ประชุมกี่โมงดี", [], "none", 0),
    ("คุณสมชาย ใจดี โทร 081-234-5678", ["has_name", "has_contact"], "contact", 2),
    ("บัตรประชาชน 1-1037-02345-67-1 ของนางสาวสมหญิง รักดี", ["has_name", "has_national_id"], "national_id", 3),
    ("เลขพัสดุ EX1234567890TH จัดส่งแล้ววันนี้", [], "none", 0),
    ("ผู้ป่วยชื่อ วิชัย แสนดี วินิจฉัยเป็นเบาหวาน", ["has_name", "has_health_or_special"], "health", 4),
    ("บ้านเลขที่ 12/3 ซอยสุขุมวิท 21 เขตวัฒนา กรุงเทพ 10110 คุณมาลี ใจงาม", ["has_name", "has_address"], "address", 2),
]


def rec(i, text, flags, cat, risk):
    qs = {"is_sensitive": {"type": "noul", "instructions": "ข้อความนี้มีข้อมูลส่วนบุคคลของบุคคลธรรมดาที่ระบุตัวตนได้หรือไม่", "label": risk > 0},
          "risk_level": {"type": "score", "instructions": "ความเสี่ยงด้านข้อมูลส่วนบุคคลระดับใด (0-4)", "criteria": RISK, "label": risk},
          "primary_category": {"type": "choice", "instructions": "ประเภทข้อมูลที่เสี่ยงสูงสุดคืออะไร",
                               "criteria": {c: c for c in CATS}, "label": cat}}
    for k, v in HAS.items():
        qs[k] = {"type": "noul", "instructions": v, "label": k in flags}
    return {"id": f"tiny-{i}", "state": text, "questions": qs}


if __name__ == "__main__":
    out = Path(__file__).parent / "fixtures"
    out.mkdir(exist_ok=True)
    with (out / "tiny_train.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(ROWS):
            f.write(json.dumps(rec(i, *r), ensure_ascii=False) + "\n")
    print("wrote", out / "tiny_train.jsonl")
