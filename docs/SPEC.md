# kev-thai-pdpa-guard — Project Spec (owned by the Senior PM, Opus)

Goal: fine-tune **Kev-4B** (`jaredpalmer/kev-4b`, LoRA + pointer head on `Qwen/Qwen3.5-4B-Base`) with **QLoRA 4-bit**
so that it, given a Thai (or Thai/English mixed) text, answers typed questions about **personal data under Thailand's
PDPA (พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562)**:

1. Is the text sensitive (contains personal data of an identifiable natural person)?
2. How risky is it, on a 0–4 scale?
3. Which categories of personal data appear?

Hardware: i5-12500H, 16 GB RAM, RTX 4060 Laptop **8 GB VRAM shared with another project's Ollama** → never assume the
GPU is free; check `nvidia-smi` first. Python **3.13** (Kev requires <3.14). Windows 11.

Repos: GitHub `Datchthana1/kev-thai-pdpa-guard` (public), HF model `Datchthana/kev-thai-pdpa-guard`, HF dataset
`Datchthana/thai-pdpa-pii-synthetic` (public). **All data is synthetic. Never include real people's data.**

## 1. Record format (Kev "labelled request", one JSON object per line)

```json
{"id": "tr-000123",
 "state": "<Thai text, or a JSON object/array>",
 "questions": {
   "is_sensitive":   {"type": "noul",   "instructions": "...", "label": true},
   "risk_level":     {"type": "score",  "instructions": "...", "criteria": ["...0", "...1", "...2", "...3", "...4"], "label": 3},
   "primary_category": {"type": "choice", "instructions": "...", "criteria": {"none": "...", "...": "..."}, "label": "national_id"},
   "has_name": {"type": "noul", "instructions": "...", "label": true},
   ...
 },
 "meta": {"genre": "line_chat", "template_id": "T-A-017", "split": "train", "pii_spans": [{"type": "national_id", "value": "1-1037-02345-67-1"}], "hard_negative": false}}
```
`meta` is ours (Kev ignores unknown top-level keys only if we strip it — the converter `scripts/to_kev.py` writes a
copy without `meta` for training).

## 2. Label taxonomy (the single source of truth — code in `src/pdpa_guard/taxonomy.py`)

PII types detected per span (`pii_spans[].type`):

| type | examples | base risk |
|---|---|---|
| `name` | ชื่อ-นามสกุลบุคคลธรรมดา (not public figures acting publicly, not company names) | 1 |
| `phone` | 08x-xxx-xxxx, +66 8x xxx xxxx | 2 |
| `email` | personal email | 2 |
| `address` | house no./moo/soi/road + sub-district/district/province/postcode | 2 |
| `dob` | วันเกิด (with year) | 1 |
| `national_id` | เลขบัตรประชาชน 13 digits **with valid checksum** | 3 |
| `passport` | passport no. | 3 |
| `bank_account` | เลขบัญชี + bank | 3 |
| `card_number` | credit/debit PAN **Luhn-valid**, optionally exp/CVV | 3 (4 with CVV or exp) |
| `credential` | password, OTP, PIN, API key/token | 4 |
| `health` | diagnosis, medication, HIV status, mental health, disability | 4 (PDPA s.26) |
| `special_category` | religion, political opinion, sexual orientation, criminal record, ethnicity/race, union membership, biometric, genetic | 4 (PDPA s.26) |
| `other_identifier` | license plate, employee/student ID, IP address, device ID, LINE ID | 1 |

### Risk score (`risk_level`, 0–4) — computed by code, not by hand
- **0 none**: no personal data of an identifiable individual (including hard negatives).
- **1 low**: only weak identifiers (a name alone, a LINE ID alone, a DOB without a name, a plate number).
- **2 medium**: contactable/locatable person: name + phone/email/address, or any phone/email/address alone.
- **3 high**: a government/financial identifier (national_id, passport, bank_account, card_number without CVV/exp).
- **4 critical**: PDPA s.26 sensitive data (health, special_category) about an identifiable person, credentials,
  full card data (PAN + exp/CVV), or national_id together with financial data (fraud kit).
Rule: `risk = max(base risks)`, then escalate +1 (cap 4) when ≥3 distinct PII types co-occur;
s.26 data with **no** identifiable person (e.g. anonymous statistics) → contributes 0.

### `primary_category` choice options (highest-risk type present; ties broken by the table order, bottom wins)
`none, name, contact, address, other_identifier, national_id, financial, credential, health, special_category`
(`contact` = phone/email; `financial` = bank_account/card_number; `national_id` includes passport; `dob` → `name` bucket
only if no other type, else ignored).

### Per-type yes/no questions (8 nouls)
`has_name, has_contact, has_address, has_national_id, has_financial, has_credential, has_health_or_special (s.26), has_other_identifier`

Total 11 questions per record. Instruction wording: Thai, with **≥4 paraphrases per question** sampled at random
(English paraphrase in 15% of records), so the model doesn't overfit one wording.

## 3. Data quality bar (quality > quantity)
Target: **train 2,400 · dev 400 · test 600** records. Composition per split:
- ~55% positive (≥1 PII), ~25% hard negatives, ~20% clean negatives.
- Hard negatives: order/tracking/invoice numbers of 13 digits that **fail** the ID checksum, 16-digit numbers that
  fail Luhn, public figures/politicians named in news, company names/addresses/phone (juristic persons are not PDPA
  personal data), fictional characters, masked data (`1-1037-xxxxx-xx-1`, `**** **** **** 1234` → risk ≤1),
  health statistics with no person, prices/dates/times.
- Genres (≥12): LINE/Messenger chat, customer-support ticket, e-commerce delivery form, HR/job application,
  hospital/clinic note, bank/loan conversation, school/university record, government form, social media post,
  email, server log / JSON payload, real-estate/rental contract, insurance claim, news article, recipe/product review.
- Realism: real Thai provinces/districts/postcodes, plausible Thai first/last names, mixed formatting
  (spaces/dashes/none, Thai digits ๑๒๓ occasionally), typos, informal Thai (ครับ/ค่ะ/นะคะ/555), code-switching.
- Validity: national IDs pass the mod-11 checksum; cards pass Luhn with correct BIN prefixes (4 Visa, 51-55 MC, 35 JCB).
- **No leakage**: test uses **held-out templates** written by a different author (template_id prefix `T-B`), dev/train
  from `T-A`. Dedupe by normalized-text MinHash/Jaccard ≥0.8 across splits.
- Every split is checked by `scripts/qa_dataset.py`: label recomputation equals stored label, class balance, length
  stats, duplicate rate, checksum validity, leak check.

## 4. Training
`third_party/kev` (official trainer) + our patch adding `--load_4bit 1` (bitsandbytes NF4, bf16 compute,
`prepare_model_for_kbit_training`). `--init_from jaredpalmer/kev-4b --base Qwen/Qwen3.5-4B-Base --lr 2e-5
--batch 1 --accum 8 --dtype bf16 --checkpointing 1 --epochs 2`. Replay of the original recipe is optional (skip if
it needs a suite we cannot download).

## 5. Evaluation (`scripts/evaluate.py`, test split, identical questions for every system)
Systems: (a) **Kev-4B base** zero-shot, (b) **Kev-4B + ours**, (c) **iapp/OpenThai-SystemOne** (via Ollama
`/v1/systemone` or transformers remote code). Metrics: per-question accuracy, `is_sensitive` P/R/F1, macro-F1 over the
8 `has_*` nouls, `primary_category` accuracy + macro-F1, `risk_level` exact accuracy, ±1 accuracy, MAE,
Brier and ECE for all questions, latency. Output `results/eval_<system>.json` + `results/comparison.md`.

## 6. Deliverables
README.md (Thai + English, explain every function, input/output, workflow between functions),
MODEL_CARD.md (= HF README), dataset card, archify architecture diagrams (HTML) under `docs/architecture/`,
push to GitHub + HF.

## Team (agent swarm)
- **Opus ×1** — Senior Project Manager (this spec, decisions, quality gates, integration, final review)
- **Sonnet ×2** — Assistant PMs: `data-lead` (dataset), `train-lead` (training + eval); docs handled by Opus + Haiku
- **Haiku ×N** — Developers spawned by the Sonnet leads, one bounded task each
