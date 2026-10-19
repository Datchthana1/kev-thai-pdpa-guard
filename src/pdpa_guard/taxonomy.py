"""Single source of truth for the PDPA-guard label taxonomy (SPEC section 2).

Public API
----------
PII_TYPES, BASE_RISK, CATEGORIES, QUESTION_IDS
effective_spans(spans)  -> spans after the "no identifiable person" rule
compute_risk(spans)     -> int 0..4
primary_category(spans) -> str
has_flags(spans)        -> dict of the 8 has_* booleans
build_questions(spans, rng, lang=None) -> dict of 11 Kev questions with labels

A span is a dict: {"type": <PII type>, "value": str, ...}. For card_number an optional
boolean ``has_cvv_or_exp`` marks full card data (risk 4).
"""
from __future__ import annotations

import random

PII_TYPES = ["name", "phone", "email", "address", "dob", "national_id", "passport", "bank_account",
             "card_number", "credential", "health", "special_category", "other_identifier"]

BASE_RISK = {"name": 1, "phone": 2, "email": 2, "address": 2, "dob": 1, "national_id": 3, "passport": 3,
             "bank_account": 3, "card_number": 3, "credential": 4, "health": 4, "special_category": 4,
             "other_identifier": 1}

# categories in table order (ties: later wins)
CATEGORIES = ["none", "name", "contact", "address", "other_identifier", "national_id", "financial",
              "credential", "health", "special_category"]

TYPE_TO_CATEGORY = {"name": "name", "phone": "contact", "email": "contact", "address": "address",
                    "dob": "name", "national_id": "national_id", "passport": "national_id",
                    "bank_account": "financial", "card_number": "financial", "credential": "credential",
                    "health": "health", "special_category": "special_category",
                    "other_identifier": "other_identifier"}

S26 = {"health", "special_category"}
# a span that lets us single out a person (needed for s.26 data to count)
IDENTIFYING = {"name", "phone", "email", "address", "national_id", "passport", "bank_account",
               "card_number", "other_identifier", "dob"}

HAS_QUESTIONS = {
    "has_name": {"name"},
    "has_contact": {"phone", "email"},
    "has_address": {"address"},
    "has_national_id": {"national_id", "passport"},
    "has_financial": {"bank_account", "card_number"},
    "has_credential": {"credential"},
    "has_health_or_special": {"health", "special_category"},
    "has_other_identifier": {"other_identifier"},
}
QUESTION_IDS = ["is_sensitive", "risk_level", "primary_category"] + list(HAS_QUESTIONS)


# ----------------------------------------------------------------------------- labels
def effective_spans(spans):
    """s.26 data about nobody (no identifying span in the text) contributes nothing."""
    if any(s["type"] in IDENTIFYING for s in spans):
        return list(spans)
    return [s for s in spans if s["type"] not in S26]


def _span_risk(s):
    r = BASE_RISK[s["type"]]
    if s["type"] == "card_number" and s.get("has_cvv_or_exp"):
        r = 4
    return r


def compute_risk(spans) -> int:
    sp = effective_spans(spans)
    if not sp:
        return 0
    risk = max(_span_risk(s) for s in sp)
    types = {s["type"] for s in sp}
    if "national_id" in types and types & {"bank_account", "card_number"}:
        risk = 4
    if len(types) >= 3:
        risk += 1
    return min(risk, 4)


def primary_category(spans) -> str:
    sp = effective_spans(spans)
    if not sp:
        return "none"
    types = {s["type"] for s in sp}
    best, key = "none", (-1, -1)
    for s in sp:
        t = s["type"]
        if t == "dob" and len(types) > 1:
            continue  # dob ignored when anything else is present
        cat = TYPE_TO_CATEGORY[t]
        k = (_span_risk(s), CATEGORIES.index(cat))
        if k > key:
            best, key = cat, k
    return best


def has_flags(spans) -> dict:
    types = {s["type"] for s in effective_spans(spans)}
    return {q: bool(types & ts) for q, ts in HAS_QUESTIONS.items()}


# ----------------------------------------------------------------------------- wording
_TH_SENS = [
    "ข้อความนี้มีข้อมูลส่วนบุคคลของบุคคลธรรมดาที่ระบุตัวตนได้หรือไม่",
    "ข้อความนี้เป็นข้อมูลอ่อนไหวตามพ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล (PDPA) หรือไม่",
    "ในข้อความนี้มีข้อมูลที่ชี้ไปถึงตัวบุคคลจริงได้หรือเปล่า",
    "ถ้านำข้อความนี้ไปเผยแพร่ จะทำให้ข้อมูลส่วนบุคคลของใครรั่วไหลหรือไม่",
    "ข้อความนี้มี PII ของบุคคลที่ระบุตัวได้ไหม",
]
_EN_SENS = [
    "Does this text contain personal data of an identifiable natural person under Thailand's PDPA?",
    "Is this text sensitive in the sense that it exposes personal information about a real individual?",
    "Would sharing this text leak personally identifiable information (PII)?",
    "Can any individual be identified from the personal data in this text?",
]
_TH_RISK = [
    "ข้อความนี้มีความเสี่ยงด้านข้อมูลส่วนบุคคลระดับใด (0-4)",
    "ให้คะแนนความเสี่ยงของการรั่วไหลข้อมูลส่วนบุคคลในข้อความนี้ จาก 0 ถึง 4",
    "หากข้อความนี้หลุดออกไป จะมีความรุนแรงต่อเจ้าของข้อมูลมากแค่ไหน",
    "ประเมินระดับความเสี่ยงตาม PDPA ของข้อความนี้",
    "ข้อมูลส่วนบุคคลในข้อความนี้อันตรายระดับไหน",
]
_EN_RISK = [
    "How risky is this text for personal-data exposure under PDPA (0-4)?",
    "Rate the privacy risk of this text from 0 (none) to 4 (critical).",
    "If this text leaked, how severe would the harm to the data subject be?",
    "Assess the PDPA risk level of this text.",
]
_TH_PRIMARY = [
    "ข้อมูลส่วนบุคคลประเภทที่เสี่ยงสูงสุดในข้อความนี้คือประเภทใด",
    "จัดข้อความนี้ในหมวดข้อมูลส่วนบุคคลที่สำคัญที่สุดหมวดใด",
    "ถ้าต้องเลือกหมวดข้อมูลหลักเพียงหมวดเดียวของข้อความนี้ คือหมวดอะไร",
    "หมวดหมู่ข้อมูลส่วนบุคคลที่มีความเสี่ยงสูงสุดที่ปรากฏในข้อความคืออะไร",
    "ข้อมูลประเภทใดในข้อความนี้ที่ควรปกป้องมากที่สุด",
]
_EN_PRIMARY = [
    "Which category of personal data is the highest-risk one present in this text?",
    "Pick the single most important personal-data category in this text.",
    "What is the primary PDPA data category of this text?",
    "Which type of data in this text most needs protecting?",
]

_HAS_TH = {
    "has_name": [
        "ข้อความนี้มีชื่อหรือนามสกุลของบุคคลธรรมดา (ไม่ใช่บริษัท/บุคคลสาธารณะที่ปรากฏเป็นข่าว) หรือไม่",
        "มีชื่อจริงหรือชื่อเล่นของบุคคลทั่วไปอยู่ในข้อความนี้ไหม",
        "ในข้อความนี้ปรากฏชื่อ-นามสกุลของบุคคลที่ระบุตัวได้หรือไม่",
        "ข้อความนี้เอ่ยชื่อบุคคลธรรมดาคนใดคนหนึ่งหรือเปล่า",
    ],
    "has_contact": [
        "ข้อความนี้มีช่องทางติดต่อส่วนบุคคล เช่น เบอร์โทรศัพท์หรืออีเมลส่วนตัว หรือไม่",
        "มีเบอร์โทรหรืออีเมลของบุคคลธรรมดาอยู่ในข้อความไหม",
        "ในข้อความนี้ปรากฏข้อมูลติดต่อ (โทรศัพท์/อีเมล) ของบุคคลหรือเปล่า",
        "ข้อความนี้เปิดเผยเบอร์มือถือหรืออีเมลของใครบางคนหรือไม่",
    ],
    "has_address": [
        "ข้อความนี้มีที่อยู่ที่พักอาศัยหรือที่อยู่จัดส่งของบุคคลหรือไม่",
        "มีที่อยู่ (บ้านเลขที่ ซอย ถนน ตำบล อำเภอ จังหวัด) ของบุคคลธรรมดาในข้อความไหม",
        "ในข้อความนี้ระบุที่อยู่ของบุคคลที่ตามตัวเจอได้หรือไม่",
        "ข้อความนี้ปรากฏที่อยู่อาศัยของใครสักคนหรือเปล่า",
    ],
    "has_national_id": [
        "ข้อความนี้มีเลขบัตรประชาชนหรือเลขหนังสือเดินทางที่ใช้ได้จริงหรือไม่",
        "มีเลขประจำตัวประชาชน 13 หลักที่ถูกต้อง หรือเลขพาสปอร์ต อยู่ในข้อความไหม",
        "ในข้อความนี้ปรากฏเลขบัตรประชาชน/พาสปอร์ตของบุคคลหรือเปล่า",
        "ข้อความนี้เปิดเผยหมายเลขเอกสารประจำตัวที่ทางราชการออกให้หรือไม่",
    ],
    "has_financial": [
        "ข้อความนี้มีข้อมูลทางการเงิน เช่น เลขบัญชีธนาคารหรือเลขบัตรเครดิต/เดบิต หรือไม่",
        "มีเลขบัญชีธนาคารหรือเลขบัตรที่ใช้งานได้จริงอยู่ในข้อความไหม",
        "ในข้อความนี้ปรากฏข้อมูลบัญชีหรือบัตรของบุคคลหรือเปล่า",
        "ข้อความนี้เปิดเผยเลขบัญชีหรือเลขบัตรชำระเงินหรือไม่",
    ],
    "has_credential": [
        "ข้อความนี้มีข้อมูลยืนยันตัวตน เช่น รหัสผ่าน OTP PIN หรือ API key หรือไม่",
        "มีรหัสผ่าน รหัส OTP หรือโทเค็นลับอยู่ในข้อความไหม",
        "ในข้อความนี้เปิดเผยรหัสเข้าใช้งานหรือรหัสลับของใครหรือเปล่า",
        "ข้อความนี้มี credential ที่นำไปเข้าระบบได้หรือไม่",
    ],
    "has_health_or_special": [
        "ข้อความนี้มีข้อมูลอ่อนไหวตามมาตรา 26 เช่น สุขภาพ ศาสนา การเมือง ประวัติอาชญากรรม ของบุคคลที่ระบุตัวได้หรือไม่",
        "มีข้อมูลสุขภาพหรือข้อมูลอ่อนไหวพิเศษ (ศาสนา เชื้อชาติ รสนิยมทางเพศ ฯลฯ) ของบุคคลในข้อความไหม",
        "ในข้อความนี้ปรากฏโรค ยา หรือข้อมูลอ่อนไหวส่วนตัวของใครหรือเปล่า",
        "ข้อความนี้เปิดเผยข้อมูลส่วนบุคคลที่มีความอ่อนไหวเป็นพิเศษหรือไม่",
    ],
    "has_other_identifier": [
        "ข้อความนี้มีตัวระบุอื่น เช่น ทะเบียนรถ รหัสพนักงาน/นักศึกษา IP LINE ID หรือไม่",
        "มีป้ายทะเบียนรถ รหัสพนักงาน รหัสนักศึกษา หรือ LINE ID ของบุคคลในข้อความไหม",
        "ในข้อความนี้ปรากฏตัวระบุตัวบุคคลแบบอื่นนอกจากชื่อและเลขบัตรหรือเปล่า",
        "ข้อความนี้เปิดเผยไอดีหรือรหัสประจำตัวของใครบางคนหรือไม่",
    ],
}
_HAS_EN = {
    "has_name": ["Does the text contain the name of a private individual (not a company or a public figure in the news)?",
                 "Is a real person's name (full name or nickname) mentioned?",
                 "Does the text name an identifiable natural person?",
                 "Is there a person's first name or surname in the text?"],
    "has_contact": ["Does the text contain a personal phone number or email address?",
                    "Is contact information (phone/email) of an individual present?",
                    "Does the text expose someone's mobile number or personal email?",
                    "Are there personal contact details in the text?"],
    "has_address": ["Does the text contain a person's home or delivery address?",
                    "Is a residential address (house no., street, district, province) present?",
                    "Does the text reveal where an individual lives?",
                    "Is there a postal address of a person in the text?"],
    "has_national_id": ["Does the text contain a valid Thai national ID or passport number?",
                        "Is a government-issued identity number of a person present?",
                        "Does the text expose a 13-digit citizen ID or passport number?",
                        "Is there an official personal ID document number in the text?"],
    "has_financial": ["Does the text contain financial data such as a bank account or card number?",
                      "Is a usable bank account number or payment card number present?",
                      "Does the text expose someone's account or card details?",
                      "Are there payment or banking details of a person in the text?"],
    "has_credential": ["Does the text contain a credential such as a password, OTP, PIN or API key?",
                       "Is a secret login code or token present?",
                       "Does the text leak anything that could be used to log in?",
                       "Are there authentication secrets in the text?"],
    "has_health_or_special": ["Does the text contain PDPA section 26 sensitive data (health, religion, politics, criminal record, etc.) about an identifiable person?",
                              "Is health information or other special-category data about a person present?",
                              "Does the text disclose a person's medical condition, medication, or other sensitive attribute?",
                              "Is there special-category personal data in the text?"],
    "has_other_identifier": ["Does the text contain other identifiers such as a license plate, employee/student ID, IP address or LINE ID?",
                             "Is a vehicle plate, staff/student number, device ID, or LINE ID present?",
                             "Does the text contain an ID handle that points to a person?",
                             "Are there other personal identifiers in the text?"],
}

_NOUL_CRIT_NOTE = None  # noul has no criteria

RISK_CRITERIA_TH = [
    "0 ไม่มี: ไม่มีข้อมูลส่วนบุคคลของบุคคลที่ระบุตัวได้",
    "1 ต่ำ: มีเพียงตัวระบุอย่างอ่อน เช่น ชื่ออย่างเดียว LINE ID วันเกิดโดยไม่มีชื่อ ทะเบียนรถ",
    "2 ปานกลาง: ติดต่อหรือระบุที่อยู่บุคคลได้ เช่น ชื่อ+เบอร์/อีเมล/ที่อยู่ หรือเบอร์/อีเมล/ที่อยู่อย่างเดียว",
    "3 สูง: เลขประจำตัวทางราชการหรือการเงิน เช่น เลขบัตรประชาชน พาสปอร์ต บัญชีธนาคาร บัตรเครดิต",
    "4 วิกฤต: ข้อมูลอ่อนไหวมาตรา 26 (สุขภาพ ฯลฯ) รหัสผ่าน/OTP ข้อมูลบัตรครบ (CVV/วันหมดอายุ) หรือเลขบัตรประชาชนคู่ข้อมูลการเงิน",
]
RISK_CRITERIA_EN = [
    "0 none: no personal data of an identifiable individual",
    "1 low: only weak identifiers (a name alone, LINE ID, DOB without name, license plate)",
    "2 medium: a contactable/locatable person (name + phone/email/address, or phone/email/address alone)",
    "3 high: government or financial identifiers (national ID, passport, bank account, card number)",
    "4 critical: PDPA s.26 sensitive data, credentials, full card data (PAN + exp/CVV), or national ID with financial data",
]
CAT_CRITERIA_TH = {
    "none": "ไม่มีข้อมูลส่วนบุคคล",
    "name": "ชื่อ-นามสกุล (หรือวันเกิด) ของบุคคล",
    "contact": "เบอร์โทรศัพท์หรืออีเมลส่วนตัว",
    "address": "ที่อยู่ของบุคคล",
    "other_identifier": "ตัวระบุอื่น เช่น ทะเบียนรถ รหัสพนักงาน/นักศึกษา IP LINE ID",
    "national_id": "เลขบัตรประชาชนหรือพาสปอร์ต",
    "financial": "เลขบัญชีธนาคารหรือบัตรเครดิต/เดบิต",
    "credential": "รหัสผ่าน OTP PIN หรือ API key",
    "health": "ข้อมูลสุขภาพ (มาตรา 26)",
    "special_category": "ข้อมูลอ่อนไหวอื่น ๆ ตามมาตรา 26 เช่น ศาสนา การเมือง ประวัติอาชญากรรม เพศวิถี เชื้อชาติ ชีวภาพ",
}
CAT_CRITERIA_EN = {
    "none": "no personal data",
    "name": "a person's name (or date of birth)",
    "contact": "personal phone number or email",
    "address": "a person's address",
    "other_identifier": "other identifiers: license plate, employee/student ID, IP, LINE ID",
    "national_id": "national ID or passport number",
    "financial": "bank account or credit/debit card number",
    "credential": "password, OTP, PIN or API key",
    "health": "health data (s.26)",
    "special_category": "other s.26 sensitive data: religion, politics, criminal record, sexual orientation, ethnicity, biometric",
}


def build_questions(spans, rng: random.Random | None = None, lang: str | None = None) -> dict:
    """Return the 11 questions (Kev format) with labels computed from ``spans``.

    lang: "th" | "en" | None (None -> English with p=0.15, else Thai; one language per record).
    """
    rng = rng or random.Random()
    if lang is None:
        lang = "en" if rng.random() < 0.15 else "th"
    en = lang == "en"
    risk = compute_risk(spans)
    flags = has_flags(spans)
    q = {
        "is_sensitive": {"type": "noul", "instructions": rng.choice(_EN_SENS if en else _TH_SENS),
                         "label": risk >= 1},
        "risk_level": {"type": "score", "instructions": rng.choice(_EN_RISK if en else _TH_RISK),
                       "criteria": list(RISK_CRITERIA_EN if en else RISK_CRITERIA_TH), "label": risk},
        "primary_category": {"type": "choice", "instructions": rng.choice(_EN_PRIMARY if en else _TH_PRIMARY),
                             "criteria": dict(CAT_CRITERIA_EN if en else CAT_CRITERIA_TH),
                             "label": primary_category(spans)},
    }
    for qid in HAS_QUESTIONS:
        q[qid] = {"type": "noul", "instructions": rng.choice((_HAS_EN if en else _HAS_TH)[qid]),
                  "label": flags[qid]}
    return q


def labels_from_spans(spans) -> dict:
    """Plain label dict {question_id: label} (used by QA and eval)."""
    risk = compute_risk(spans)
    out = {"is_sensitive": risk >= 1, "risk_level": risk, "primary_category": primary_category(spans)}
    out.update(has_flags(spans))
    return out
