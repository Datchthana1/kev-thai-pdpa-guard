"""Create value packs for free-form (non-template) long records.

Each pack lists fake values (from pdpa_guard.fakers) that a writer must weave into a natural Thai text of
300-1200 chars. Labels are computed later from the pack's spans, never by the writer.

Usage: python scripts/make_value_packs.py [--seed 7]   -> data/longform/packs_{A,B}/pack_NN.json
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdpa_guard import fakers  # noqa: E402

ROLE = {
    "name": "ชื่อ-นามสกุลของบุคคลหลัก", "name2": "ชื่อ-นามสกุลของบุคคลอีกคน (เช่น ญาติ เพื่อน เจ้าหน้าที่)",
    "first_name": "ชื่อต้น/ชื่อเล่นของบุคคล", "phone": "เบอร์โทรของบุคคลหลัก", "phone2": "เบอร์โทรของบุคคลอีกคน",
    "email": "อีเมลส่วนตัวของบุคคลหลัก", "address": "ที่อยู่ของบุคคลหลัก", "dob": "วันเกิด",
    "national_id": "เลขบัตรประชาชนของบุคคลหลัก", "passport": "เลขหนังสือเดินทาง",
    "bank_account": "บัญชีธนาคาร (มีชื่อธนาคารรวมอยู่)", "card": "เลขบัตรเครดิต/เดบิต", "card_full": "ข้อมูลบัตรครบ (เลขบัตร+วันหมดอายุ+CVV)",
    "password": "รหัสผ่าน", "otp": "รหัส OTP", "pin": "รหัส PIN", "api_key": "API key / token",
    "health": "ข้อมูลสุขภาพของบุคคลหลัก (ใช้เป็นข้อเท็จจริงเกี่ยวกับเขา/เธอ)",
    "special": "ข้อมูลอ่อนไหว (ศาสนา/การเมือง/ประวัติอาชญากรรม/เพศวิถี/เชื้อชาติ ฯลฯ) ของบุคคลหลัก",
    "plate": "ทะเบียนรถ", "employee_id": "รหัสพนักงาน", "student_id": "รหัสนักศึกษา", "ip": "IP address",
    "line_id": "LINE ID",
    "public_figure": "บุคคลสาธารณะ (ใช้ในบริบทสาธารณะ เป็นกลาง)", "company": "ชื่อบริษัท/นิติบุคคล",
    "company_phone": "เบอร์โทรของบริษัท", "company_email": "อีเมลของบริษัท", "company_address": "ที่อยู่สำนักงานบริษัท",
    "health_stat": "สถิติสุขภาพภาพรวม (ไม่เกี่ยวกับบุคคลใด)", "special_stat": "สถิติภาพรวม (ศาสนา/ประชากร ฯลฯ ไม่เกี่ยวกับบุคคลใด)",
    "order_no": "เลขคำสั่งซื้อ (หน้าตาคล้ายเลขบัตร แต่ไม่ใช่)", "tracking_no": "เลขพัสดุ", "invoice_no": "เลขใบแจ้งหนี้",
    "fake_card": "เลข 16 หลัก (ไม่ใช่บัตรจริง เช่น เลขอ้างอิง)", "masked_id": "เลขบัตรที่ถูกปิดบัง", "masked_card": "เลขบัตรที่ถูกปิดบัง",
    "masked_phone": "เบอร์ที่ถูกปิดบัง", "masked_account": "เลขบัญชีที่ถูกปิดบัง", "url": "เว็บไซต์", "date": "วันที่",
    "amount": "จำนวนเงิน (บาท)", "price": "ราคา", "time": "เวลา", "minutes": "จำนวนนาที", "workdays": "จำนวนวันทำการ",
    "people": "จำนวนคน", "percent": "ร้อยละ", "province": "จังหวัด", "case_no": "เลขที่เรื่อง/เคส", "menu_item": "เมนู",
    "gadget": "สินค้าอิเล็กทรอนิกส์", "grocery": "ของใช้/อาหาร", "service_name": "ชื่อบริการ", "months": "จำนวนเดือน",
    "years": "จำนวนปี", "rating": "คะแนนรีวิว",
}

# profiles of PII keys (first element is always the "main" identity if any)
P_LOW = [["name"], ["line_id"], ["plate"], ["dob", "name"], ["name", "employee_id"], ["name", "student_id"], ["name", "plate"],
         ["ip"], ["first_name", "line_id"]]
P_MID = [["name", "phone"], ["name", "email"], ["name", "address"], ["name", "phone", "email"], ["name", "phone", "address"],
         ["phone"], ["email"], ["name", "phone", "name2", "phone2"], ["name", "email", "address", "dob"]]
P_HIGH = [["name", "national_id", "address"], ["name", "bank_account", "phone"], ["name", "passport", "email"],
          ["name", "card", "phone"], ["name", "national_id", "dob", "address"], ["name", "bank_account", "national_id"],
          ["name", "passport", "phone", "dob"]]
P_CRIT = [["name", "health", "phone"], ["name", "health", "dob", "address"], ["name", "special", "phone"],
          ["name", "special"], ["name", "password", "email"], ["name", "otp", "phone"], ["name", "card_full", "phone"],
          ["name", "national_id", "bank_account", "phone"], ["name", "health", "national_id", "address", "name2"],
          ["name", "pin", "bank_account"], ["email", "api_key", "ip"], ["name", "special", "name2", "phone2"],
          ["name", "health", "name2", "phone2"], ["name", "api_key", "employee_id"]]
GENRE_PREF = {
    "hospital_note": P_CRIT[:3] + P_CRIT[8:9] + P_CRIT[12:13],
    "bank_loan": [P_HIGH[1], P_HIGH[3], P_HIGH[5], P_CRIT[4], P_CRIT[5], P_CRIT[6], P_CRIT[7], P_CRIT[9]],
    "server_log": [["email", "api_key", "ip"], ["name", "email", "ip"], ["name", "national_id", "phone"], ["ip", "email"],
                   ["name", "password", "email"], ["name", "card_full"]],
    "insurance_claim": [P_CRIT[0], P_CRIT[1], P_HIGH[0], P_HIGH[1], ["name", "plate", "phone"], ["name", "health", "bank_account"]],
    "school_record": [["name", "student_id", "phone"], ["name", "student_id", "special"], ["name", "student_id", "health", "name2"],
                      ["name", "dob", "address", "name2", "phone2"]],
    "hr_application": [P_HIGH[0], P_HIGH[2], P_MID[8], P_HIGH[6], ["name", "employee_id", "special"], ["name", "phone", "email", "bank_account"]],
    "realestate_contract": [P_HIGH[0], ["name", "national_id", "address", "name2", "national_id2"] , ["name", "bank_account", "address", "name2", "phone2"]],
    "gov_form": [P_HIGH[0], P_HIGH[4], P_HIGH[6], ["name", "national_id", "address", "phone"]],
    "line_chat": [P_MID[0], P_MID[3], P_LOW[1], P_HIGH[1], P_CRIT[5], ["name", "bank_account"], ["name", "phone", "name2"]],
    "ecommerce_delivery": [P_MID[4], P_MID[5], P_MID[2], ["name", "phone", "address", "card"]],
    "support_ticket": [P_MID[1], P_MID[0], P_CRIT[4], P_CRIT[5], P_HIGH[3]],
    "email": [P_MID[3], P_HIGH[0], P_CRIT[4], P_MID[1], ["name", "email", "phone", "address", "name2"]],
    "social_post": [P_LOW[0], P_LOW[1], P_MID[0], ["name", "special"], ["name", "health"], P_LOW[6]],
    "news_article": [P_LOW[0], ["name", "address", "phone"], ["name", "special"], ["name", "health", "name2"]],
    "review": [P_LOW[1], P_MID[0], P_LOW[0], P_MID[1]],
}
HN_POOL = {
    "default": ["public_figure", "company", "company_phone", "company_email", "company_address", "health_stat", "special_stat",
                "order_no", "tracking_no", "invoice_no", "fake_card", "masked_id", "masked_card", "masked_phone", "masked_account",
                "url", "date", "amount"],
    "news_article": ["public_figure", "company", "health_stat", "special_stat", "date", "company_address", "percent", "amount"],
    "ecommerce_delivery": ["order_no", "tracking_no", "invoice_no", "fake_card", "masked_card", "masked_phone", "company", "company_phone", "amount", "date"],
    "bank_loan": ["fake_card", "masked_card", "masked_account", "masked_id", "company", "company_phone", "amount", "invoice_no", "date"],
    "hospital_note": ["health_stat", "company", "company_phone", "masked_id", "percent", "date"],
    "server_log": ["order_no", "tracking_no", "invoice_no", "fake_card", "masked_phone", "masked_id", "url", "date", "case_no"],
}
CLEAN_POOL = ["date", "amount", "minutes", "workdays", "people", "percent", "province", "menu_item", "gadget", "grocery",
              "service_name", "months", "years", "rating", "company", "url", "time", "case_no"]
FORMATS = {
    "line_chat": ["บทสนทนา LINE กลุ่มหรือส่วนตัว ยาว 12-25 ข้อความ มีชื่อผู้พูดนำหน้า/เวลา", "แชตกับร้านค้าหรือแอดมินหลายข้อความ"],
    "support_ticket": ["ตั๋วซัพพอร์ตเต็มรูปแบบ มีหัวข้อ รายละเอียดปัญหา ขั้นตอนที่ลองแล้ว และบันทึกของเจ้าหน้าที่", "บันทึกการคุยโทรศัพท์ของ call center"],
    "ecommerce_delivery": ["ฟอร์มสั่งซื้อ/ใบจัดส่งหลายบรรทัด พร้อมรายการสินค้า", "อีเมลยืนยันคำสั่งซื้อฉบับเต็ม"],
    "hr_application": ["เรซูเม่/ใบสมัครงานแบบฟอร์มหลายหัวข้อ", "อีเมลสมัครงานพร้อมจดหมายปะหน้า", "บันทึกสัมภาษณ์/ตรวจสอบประวัติของ HR"],
    "hospital_note": ["สรุปการจำหน่ายผู้ป่วย (discharge summary) หลายหัวข้อ", "บันทึกพยาบาล/แพทย์ประจำวัน", "ใบส่งตัว/ใบนัดหมาย"],
    "bank_loan": ["แบบฟอร์มคำขอสินเชื่อหลายหัวข้อ", "บทสนทนาเจ้าหน้าที่ธนาคารกับลูกค้าทางโทรศัพท์/แชต", "อีเมลแจ้งผลอนุมัติ"],
    "school_record": ["ใบรายงานผล/บันทึกครูที่ปรึกษา หลายหัวข้อ", "จดหมายแจ้งผู้ปกครองฉบับเต็ม", "บันทึกการให้คำปรึกษานักเรียน"],
    "gov_form": ["แบบคำร้องหน่วยงานรัฐ มีช่องกรอกหลายช่องและคำชี้แจง", "หนังสือราชการ/บันทึกข้อความ"],
    "social_post": ["โพสต์ยาวในกลุ่ม Facebook / Pantip พร้อมคอมเมนต์ตอบกลับหลายอัน", "เธรดทวีตหลายโพสต์"],
    "email": ["อีเมลฉบับเต็ม มี From/To/Subject ทักทาย เนื้อหาหลายย่อหน้า ลงท้าย ลายเซ็น", "อีเมลต่อเนื่อง (reply chain) 2-3 ฉบับ"],
    "server_log": ["JSON object ของ log/payload มีหลายฟิลด์ (>=8 ฟิลด์) และอาจซ้อนกัน (ให้ state เป็น JSON object ไม่ใช่สตริง)",
                   "ไฟล์ log ข้อความหลายบรรทัด (>=8 บรรทัด) แบบแอปพลิเคชัน/nginx", "JSON array ของเหตุการณ์ audit หลายรายการ (state เป็น JSON)"],
    "realestate_contract": ["สัญญาเช่า/ซื้อขายมีข้อตกลงหลายข้อ คู่สัญญา และลงชื่อพยาน", "บันทึกข้อตกลงเช่าห้องพัก/คอนโด"],
    "insurance_claim": ["แบบคำขอสินไหมหลายหัวข้อ พร้อมคำบรรยายเหตุการณ์", "จดหมายตอบรับ/ปฏิเสธเคลม"],
    "news_article": ["ข่าวยาว 3-5 ย่อหน้า มีพาดหัว", "บทความสถิติ/วิเคราะห์ภาพรวม"],
    "review": ["รีวิวร้านอาหาร/โรงแรม/สินค้ายาว หลายย่อหน้า", "สูตรอาหารพร้อมวิธีทำ"],
}
GENRES = list(FORMATS)


def draw_values(keys, rng):
    vals, spans_all = [], []
    cache = {}
    for k in keys:
        base = re.sub(r"\d+$", "", k)
        if k == "national_id2":
            base = "national_id"
        text, spans = fakers.gen_slot(base, rng)
        vals.append({"key": k, "role": ROLE.get(k, ROLE.get(base, base)), "value": text,
                     "pii_type": spans[0]["type"] if spans else None})
        spans_all.extend(spans)
    return vals, spans_all


def make_pack(pid, genre, rng):
    r = rng.random()
    kind = "positive" if r < 0.60 else ("hard_negative" if r < 0.88 else "clean_negative")
    if kind == "positive":
        if rng.random() < 0.65 and genre in GENRE_PREF:
            keys = rng.choice(GENRE_PREF[genre])
        else:
            keys = rng.choice(rng.choice([P_LOW, P_MID, P_MID, P_HIGH, P_HIGH, P_CRIT, P_CRIT, P_CRIT]))
        keys = list(keys)
    elif kind == "hard_negative":
        pool = HN_POOL.get(genre, HN_POOL["default"])
        keys = rng.sample(pool, rng.randint(3, min(6, len(pool))))
        lookalike = [k for k in keys if k in ("order_no", "tracking_no", "invoice_no", "fake_card", "masked_id", "masked_card",
                                               "masked_phone", "masked_account", "public_figure", "company_phone", "health_stat", "special_stat")]
        if not lookalike:
            keys[0] = rng.choice(["order_no", "masked_card", "public_figure", "health_stat"])
    else:
        keys = rng.sample(CLEAN_POOL, rng.randint(2, 5))
    keys = list(dict.fromkeys(keys))
    vals, spans = draw_values(keys, rng)
    fmt = rng.choice(FORMATS[genre])
    return {"pack_id": pid, "genre": genre, "kind": kind, "format": fmt, "target_chars": rng.choice([300, 450, 600, 800, 1000, 1200]),
            "values": vals, "spans": spans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for side, files, per in (("A", 8, 50), ("B", 3, 40)):
        # note: distinct seeds so B values never repeat A's
        r = random.Random(f"{a.seed}-{side}")
        out = ROOT / "data" / "longform" / f"packs_{side}"
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in range(1, files + 1):
            packs = []
            for i in range(per):
                n += 1
                packs.append(make_pack(f"{side}{n:04d}", GENRES[(n - 1) % len(GENRES)], r))
            (out / f"pack_{f:02d}.json").write_text(json.dumps(packs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(side, n, "packs")


if __name__ == "__main__":
    main()
