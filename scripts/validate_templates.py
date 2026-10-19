"""Validate template JSONL files. Usage: python validate_templates.py file1.jsonl [file2 ...]  (exit 1 on error)."""
import json
import re
import sys

GENRES = {"line_chat", "support_ticket", "ecommerce_delivery", "hr_application", "hospital_note", "bank_loan",
          "school_record", "gov_form", "social_post", "email", "server_log", "realestate_contract",
          "insurance_claim", "news_article", "review"}
PII = {"name", "first_name", "phone", "email", "address", "dob", "national_id", "passport", "bank_account", "card",
       "card_full", "password", "otp", "pin", "api_key", "health", "special", "plate", "employee_id", "student_id",
       "ip", "line_id"}
NOPII = {"order_no", "tracking_no", "invoice_no", "fake_card", "masked_id", "masked_card", "masked_phone",
         "masked_account", "public_figure", "fictional", "company", "company_phone", "company_address",
         "company_email", "url", "health_stat", "special_stat", "price", "date", "time", "product", "percent",
         "number", "province", "quantity", "case_no"}
IDENT = PII - {"health", "special", "password", "otp", "pin", "api_key"}
NOPII |= {"minutes", "hours", "days", "workdays", "weeks", "months", "years", "age", "kid_age", "people", "seats",
          "count", "mm", "temp", "items", "pages", "floor", "km", "score", "views", "rating", "amount", "menu_item",
          "food_item", "drink_item", "dessert", "gadget", "grocery", "stationery", "clothing", "otc_medicine",
          "service_name"}
SLOT_RE = re.compile(r"\{([a-z_]+?)(\d*)\}")


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def main(paths):
    errs, n, seen = [], 0, set()
    for p in paths:
        for i, line in enumerate(open(p, encoding="utf-8"), 1):
            if not line.strip():
                continue
            n += 1
            where = f"{p}:{i}"
            try:
                t = json.loads(line)
            except Exception as e:
                errs.append(f"{where} bad json {e}")
                continue
            tid = t.get("template_id")
            if tid in seen:
                errs.append(f"{where} duplicate id {tid}")
            seen.add(tid)
            if t.get("genre") not in GENRES:
                errs.append(f"{where} bad genre {t.get('genre')}")
            kind = t.get("kind")
            if kind not in ("positive", "hard_negative", "clean_negative"):
                errs.append(f"{where} bad kind {kind}")
            if ("text" in t) == ("json" in t):
                errs.append(f"{where} need exactly one of text/json")
                continue
            body = list(strings(t["text"] if "text" in t else t["json"]))
            joined = "\n".join(body)
            if "text" in t:
                stripped = SLOT_RE.sub("", joined)
                if "{" in stripped or "}" in stripped:
                    errs.append(f"{where} stray brace")
            slots = {m.group(1) for m in SLOT_RE.finditer(joined)}
            bad = slots - PII - NOPII
            if bad:
                errs.append(f"{where} unknown slots {bad}")
            used_pii, used_id = slots & PII, slots & IDENT
            if kind == "positive" and not used_pii:
                errs.append(f"{where} positive without PII slot")
            if kind != "positive" and used_pii:
                errs.append(f"{where} {kind} uses PII slots {used_pii}")
            if kind == "positive" and used_pii and not used_id and (used_pii & {"health", "special"}):
                errs.append(f"{where} s.26 slot without identifying slot")
            if re.search(r"\d{9,}", SLOT_RE.sub("", joined)):
                errs.append(f"{where} literal long digit string")
    for e in errs:
        print("ERR", e)
    print(f"{n} templates, {len(errs)} errors")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
