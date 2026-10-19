---
license: apache-2.0
language:
  - th
  - en
pretty_name: Thai PDPA PII (synthetic)
size_categories:
  - 1K<n<10K
task_categories:
  - text-classification
  - token-classification
tags:
  - pdpa
  - pii
  - privacy
  - thai
  - synthetic
  - decision-model
  - kev
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: validation
        path: dev.jsonl
      - split: test
        path: test.jsonl
---

# Thai PDPA PII: synthetic decision dataset

**3,926 synthetic Thai (and Thai/English mixed) texts, each labelled with 11 questions about personal data under
Thailand's Personal Data Protection Act B.E. 2562 (PDPA)**: is the text sensitive, how risky (0–4), which category is
the riskiest, and which of 8 data types appear. Built to fine-tune
[`Datchthana/kev-thai-pdpa-guard`](https://huggingface.co/Datchthana/kev-thai-pdpa-guard).

> **Every person, ID number, card number, account and address in this dataset is fake.** National IDs are generated
> with a valid checksum and card numbers pass Luhn so the data is realistic, but none belongs to a real person.

## Splits

| Split | Records | Positive | Hard negative | Clean negative | Free-form long texts | Templates |
|---|---|---|---|---|---|---|
| train | 2,741 | 55.9% | 25.1% | 19.0% | 341 | 399 (A) |
| validation (dev) | 450 | 56.2% | 25.3% | 18.4% | 50 | 71 (A, disjoint from train) |
| test | 735 | 57.4% | 24.1% | 18.5% | 115 | 216 (B, different authors) |

Risk distribution (0/1/2/3/4): train 1210/299/335/288/609 · dev 197/29/48/73/103 · test 313/55/60/87/220.
15 genres in every split: LINE chat, support ticket, e-commerce delivery, HR application, hospital note, bank/loan,
school record, government form, social post, email, server log/JSON, rental contract, insurance claim, news, review.

## Record format

`train.jsonl` / `dev.jsonl` / `test.jsonl` (one JSON object per line):

```json
{"id": "te-000123",
 "state": "<the text, or a JSON object for log/form genres>",
 "questions": {
   "is_sensitive":     {"type": "noul",   "instructions": "...", "label": true},
   "risk_level":       {"type": "score",  "instructions": "...", "criteria": ["0 ...", "1 ...", "2 ...", "3 ...", "4 ..."], "label": 4},
   "primary_category": {"type": "choice", "instructions": "...", "criteria": {"none": "...", "name": "...", "...": "..."}, "label": "health"},
   "has_name": {"type": "noul", "instructions": "...", "label": true},
   "...": "8 has_* questions in total"
 },
 "meta": {"genre": "hospital_note", "template_id": "T-B-067", "kind": "positive",
          "pii_spans": [{"type": "name", "value": "..."}, {"type": "health", "value": "..."}]}}
```

`kev/*.jsonl` holds the same records without `meta`: exactly the request shape that
[Kev](https://github.com/jaredpalmer/kev) trains on (`python -m kev.train --data kev/train.jsonl`).
`pii_spans` makes the dataset usable for span detection too.

## Label rules (computed by code, never by hand)

All labels are derived from `meta.pii_spans` by `pdpa_guard.taxonomy` in the GitHub repo:

| Risk | Meaning |
|---|---|
| 0 | No personal data of an identifiable person (incl. hard negatives) |
| 1 | Weak identifier only: name alone, LINE ID, plate, DOB without name |
| 2 | Contactable/locatable: phone, email or address (with or without name) |
| 3 | Government or financial identifier: national ID, passport, bank account, card number |
| 4 | PDPA s.26 sensitive data about an identifiable person (health, religion, politics, sexual orientation, criminal record, ethnicity, biometrics), credentials (password/OTP/PIN/API key), full card data (number + expiry/CVV), or national ID together with financial data |

Plus: +1 (max 4) when three or more distinct PII types co-occur; s.26 data with no identifiable person counts as 0.
`primary_category` ∈ {none, name, contact, address, other_identifier, national_id, financial, credential, health,
special_category} is the highest-risk category present.

## How it was built

1. **Fake values**: `fakers.py` (48 slot generators) gives Thai names, real provinces/districts/postcodes, mod-11
   national IDs, Luhn cards with real BIN prefixes, Thai bank accounts, plates, LINE IDs, health conditions and s.26 phrases,
   plus look-alikes that fail the checks (order numbers, invalid cards, masked values) for hard negatives.
2. **Templates**: 686 slot templates written by separate Claude Haiku agents: 470 "A" templates for train/dev and 216 "B"
   templates for test by authors who never saw the A set.
3. **Free-form long texts**: 506 texts of 300–1,200 characters written by 11 Haiku agents around pre-generated value
   packs; a record is kept only if every PII value appears verbatim and no extra phone/email/ID/sensitive keyword leaked in.
4. **QA gate** (`qa_dataset.py`, must exit 0): label recomputation, checksum/Luhn validity, no template leakage,
   0 cross-split near-duplicates (char 5-gram Jaccard ≥ 0.8), class balance report. Splits are frozen by sha256 (`FROZEN.sha256`).
5. Question instructions have ≥4 Thai paraphrases each; 15% of records use English paraphrases.

## Known limitations

- Synthetic text: realistic but not real-world; OCR noise, very long documents and heavy slang are under-represented.
- Some long-form records pair a condition with an implausible person (e.g. a sex-specific condition); labels are still correct.
- Test `primary_category = name` has only 14 records.
- Risk levels are this project's operationalisation of PDPA principles, not an official PDPC standard.

## License

Apache-2.0. Generated with Claude (Anthropic) agents; contains no real personal data.
