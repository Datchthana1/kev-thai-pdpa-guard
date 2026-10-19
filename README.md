# kev-thai-pdpa-guard

**A model that checks whether Thai text contains personal data under the PDPA, and how risky it is**
Fine-tuned from [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) (a Jev-style decision model) with QLoRA 4-bit on a single laptop GPU (RTX 4060, 8 GB)

[![Model](https://img.shields.io/badge/🤗%20Model-Datchthana%2Fkev--thai--pdpa--guard-yellow)](https://huggingface.co/Datchthana/kev-thai-pdpa-guard)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-thai--pdpa--pii--synthetic-orange)](https://huggingface.co/datasets/Datchthana/thai-pdpa-pii-synthetic)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

> ⚠️ **All data in this project is synthetic.** The names, ID numbers, card numbers and addresses are fake, and none of them belongs to a real person.
> The model is a screening aid, not a legal opinion. Final decisions under the PDPA must be reviewed by a person.

---

## Contents
1. [What this model does (in short)](#1-what-this-model-does-in-short)
2. [Quick start](#2-quick-start)
3. [The 11 questions the model answers](#3-the-11-questions-the-model-answers)
4. [Results and comparison](#4-results-and-comparison)
5. [System architecture](#5-system-architecture)
6. [Every function: what it does, input → output](#6-every-function-what-it-does-input--output)
7. [Workflows between functions](#7-workflows-between-functions)
8. [Dataset](#8-dataset)
9. [How we trained it](#9-how-we-trained-it)
10. [Limitations](#10-limitations)
11. [How the team worked (agent swarm)](#11-how-the-team-worked-agent-swarm)

---

## 1. What this model does (in short)

You give it **any text** (a chat message, an email, a form, a JSON log, a hospital note), and it answers in one go:

| Question | Example answer |
|---|---|
| Does it contain personal data? | `True` (probability 0.98) |
| How risky is it (0–4)? | `4 = critical` |
| Which category is the riskiest? | `health` (health data, PDPA section 26) |
| Which types of data appear? | `has_name`, `has_contact`, `has_health_or_special` |

What sets it apart from an ordinary LLM: **it doesn't generate text.** It reads the text once (a single forward pass) and returns a **probability** for each option, so there's no free-form answer to parse and nothing to hallucinate. You can set a threshold directly, e.g. "if p_sensitive > 0.9, block automatically; if 0.5–0.9, send to a person for review".

---

## 2. Quick start

### Install
```bash
git clone https://github.com/Datchthana1/kev-thai-pdpa-guard.git
cd kev-thai-pdpa-guard
bash scripts/setup_env.sh          # creates .venv (Python 3.13), installs torch cu128 + Kev + bitsandbytes
```
You need an NVIDIA GPU with ≥ 6 GB (it uses about 3–4.5 GB in 4-bit mode). Python 3.12–3.13.

### Use it from Python
```python
import sys; sys.path.insert(0, "src")
from pdpa_guard.guard import PDPAGuard

guard = PDPAGuard("Datchthana/kev-thai-pdpa-guard")   # downloads the adapter from HF and loads the 4-bit backbone
r = guard.check("รบกวนโอนคืนที่บัญชี กสิกรไทย 123-4-56789-0 ชื่อบัญชี สมชาย ใจดี โทร 081-234-5678 ค่ะ")

r.is_sensitive        # True
r.p_sensitive         # 0.99
r.risk_level          # 3
r.risk_label          # 'high'
r.primary_category    # 'financial'
r.detected()          # ['has_name', 'has_contact', 'has_financial']
r.to_dict()           # everything as a dict (including the probabilities of every option)
```

### Use it from the command line
```bash
PYTHONPATH=src python -m pdpa_guard.guard "ผู้ป่วย นายวิชัย ศรีสุข เป็นเบาหวาน ต้องฉีดอินซูลินวันละ 2 ครั้ง"
# sensitive : True (p=0.997)
# risk      : 4 critical / วิกฤต (expected 3.91)
# category  : health (0.962)
# detected  : has_name, has_health_or_special
```
Add `--json` to get JSON output, or `--model runs/pdpa-4b` to use a checkpoint you trained yourself.

---

## 3. The 11 questions the model answers

Every call asks the same 11 questions (defined in [`src/pdpa_guard/taxonomy.py`](src/pdpa_guard/taxonomy.py)):

| # | Question ID | Type | Meaning | Answer |
|---|---|---|---|---|
| 1 | `is_sensitive` | `noul` (yes/no) | Does it contain personal data of an identifiable person? | p(yes) |
| 2 | `risk_level` | `score` (0–4) | Risk level | Probability for each level + expected value |
| 3 | `primary_category` | `choice` (10 options) | The highest-risk category present | Probability for each option |
| 4 | `has_name` | `noul` | First/last name of a natural person | p(yes) |
| 5 | `has_contact` | `noul` | Personal phone number / email | p(yes) |
| 6 | `has_address` | `noul` | Home address | p(yes) |
| 7 | `has_national_id` | `noul` | National ID / passport number | p(yes) |
| 8 | `has_financial` | `noul` | Bank account / credit or debit card number | p(yes) |
| 9 | `has_credential` | `noul` | Password, OTP, PIN, API key | p(yes) |
| 10 | `has_health_or_special` | `noul` | Sensitive data under section 26 (health, religion, politics, criminal record, sexual orientation, ethnicity, biometrics) | p(yes) |
| 11 | `has_other_identifier` | `noul` | License plate, employee/student ID, IP, LINE ID | p(yes) |

### Risk levels (`risk_level`)

| Level | Name | Example | Rule |
|---|---|---|---|
| **0** | None | Company news, anonymous health statistics, a masked number `**** 1234` | No data about an identifiable person |
| **1** | Low | A name alone, a LINE ID alone, a license plate | Weak identifiers |
| **2** | Medium | Name + phone, or a home address | The person can be contacted or located |
| **3** | High | National ID number, passport, bank account, card number | Government or financial identifiers |
| **4** | Critical | A named person's illness, OTP/password, full card details with CVV, ID number + financial data | Section 26 data, credentials, or a combination usable for fraud |

> Escalation rule: if **3 or more distinct types** appear together, the level goes up by 1 (capped at 4), because combined data is riskier.
> Section 26 data that **can't be linked to anyone** (e.g. "51% of diabetes patients...") counts as 0.

---

## 4. Results and comparison

<!-- RESULTS:START -->
Tested on the **test set of 735 records** (written by different authors from the training data, never seen during training). All 3 systems received **identical text + questions** (verified by an `input_sha` hash on every record):

- **kev-base**: `jaredpalmer/kev-4b` before fine-tuning (zero-shot)
- **kev-pdpa**: this model (fine-tuned)
- **openthai**: [`iapp/OpenThai-SystemOne`](https://huggingface.co/iapp/OpenThai-SystemOne) (0.8B, a Thai Jev-style decision model)

### Headline metrics

| Metric | Kev-4B base | **kev-thai-pdpa-guard** | OpenThai-SystemOne | Change vs base |
|---|---|---|---|---|
| is_sensitive F1 | 0.890 | **0.990** | 0.442 | +0.100 |
| is_sensitive precision / recall | 0.876 / 0.905 | **0.993 / 0.988** | 0.968 / 0.287 | |
| has_* macro-F1 (8 types) | 0.649 | **0.934** | 0.620 | +0.285 |
| primary_category accuracy | 0.638 | **0.946** | 0.463 | +30.8 pp |
| primary_category macro-F1 | 0.565 | **0.931** | 0.436 | +0.366 |
| risk_level exact accuracy | 0.476 | **0.912** | 0.524 | **+43.6 pp** |
| risk_level ±1 accuracy | 0.703 | **0.977** | 0.702 | +27.4 pp |
| risk_level MAE (↓) | 1.033 | **0.125** | 1.083 | −88% |
| Brier, all questions (↓) | 0.247 | **0.040** | 0.295 | −84% |
| ECE, calibration (↓) | 0.032 | **0.011** | 0.063 | −66% |
| Latency per text (RTX 4060, 4-bit) | 1.31 s | 1.28 s | **0.12 s** | |

![headline](results/bar_headline.png)

### Accuracy per question

| Question | base | **pdpa-guard** | OpenThai |
|---|---|---|---|
| is_sensitive | 0.872 | **0.989** | 0.585 |
| risk_level | 0.476 | **0.912** | 0.524 |
| primary_category | 0.638 | **0.946** | 0.463 |
| has_name | 0.959 | **0.969** | 0.824 |
| has_contact | 0.927 | **0.992** | 0.917 |
| has_address | 0.941 | **0.996** | 0.944 |
| has_national_id | 0.966 | **0.997** | 0.946 |
| has_financial | 0.863 | **0.993** | 0.899 |
| has_credential | 0.811 | **0.992** | 0.898 |
| has_health_or_special (s.26) | 0.718 | **0.985** | 0.884 |
| has_other_identifier | 0.824 | **0.947** | 0.878 |

### Accuracy by true risk level

| True risk | n | base | **pdpa-guard** | OpenThai |
|---|---|---|---|---|
| 0 none | 313 | 0.709 | **0.981** | 0.840 |
| 1 low | 55 | 0.327 | **0.927** | 0.200 |
| 2 medium | 60 | **0.983** | 0.917 | 0.650 |
| 3 high | 87 | 0.195 | **0.667** | 0.034 |
| 4 critical | 220 | 0.155 | **0.905** | 0.314 |

### Reading the results
- **Fine-tuning improved the model a lot on every metric.** The biggest gain is risk_level (47.6% → 91.2%): base Kev understands questions generally but doesn't know the PDPA risk rules, especially level 4 (section 26 data), where it scored only 15.5% and now scores 90.5%.
- **Hard negatives** (data that looks sensitive but isn't): exact risk accuracy is 0.966, versus 0.503 for base. The model learned to tell an order number from an ID number, and a public figure from a private individual.
- **Calibration got better too** (ECE 0.011). When the model says 0.9, it's right about 90% of the time, so you can set thresholds directly.
- **OpenThai-SystemOne** is very fast (0.12 s, 10× faster) and decent on the individual has_* questions, but has very low is_sensitive recall (0.287): it misses about 71% of the sensitive texts. Its answers are also internally inconsistent, e.g. risk 4 but "not sensitive". It wasn't trained for PDPA (this is zero-shot), so the comparison shows what domain fine-tuning adds, not that OpenThai is a bad model.
- **Remaining weak spots:** risk 3 (high), at 0.667. Of the 29 misses, 23 were predicted **lower** (mostly as 2), and almost all are cases of the **"3 or more types → +1"** escalation rule, e.g. name + phone + address, whose base is 2 but which escalates to 3. The model hasn't fully learned this counting rule yet. Training on more examples of it, or computing the escalation in code after the model answers, would help. Also risk 2, where it's slightly below base (0.917 vs 0.983); base tends to answer 2 for almost everything, so it happens to score well on this level.
- ⚠️ The test set comes from the same data generator family (different authors and templates), so these absolute numbers are **an optimistic estimate** of real-world use. The part to trust is the **comparison between systems on identical records**.

More detail (by genre, calibration by question type, confusion matrix): [`results/comparison.md`](results/comparison.md) · charts: `results/risk_confusion_*.png`, `results/reliability_*.png`, `results/bar_by_genre.png`
<!-- RESULTS:END -->

---

## 5. System architecture

The diagrams were made with the [archify](https://github.com/tt-a1i/archify) skill. Each one is an HTML file you can open in a browser (zoom, dark/light theme, export PNG/SVG):

| Diagram | File | What it shows |
|---|---|---|
| System architecture | [`docs/architecture/system-architecture.html`](docs/architecture/system-architecture.html) | The full picture: data pipeline → QLoRA training on the GPU → PDPAGuard → evaluation |
| Sequence of `check()` | [`docs/architecture/check-sequence.html`](docs/architecture/check-sequence.html) | Step by step, what happens inside one call |
| Dataset dataflow | [`docs/architecture/dataset-dataflow.html`](docs/architecture/dataset-dataflow.html) | How each record is built and where the labels come from |

Textual summary of the overall architecture:

```
┌────────────────────── Dataset pipeline (CPU) ───────────────────────┐
│ fakers.py ─┐                         taxonomy.py (PDPA rules)        │
│            ▼                                 ▼                       │
│ Templates ─► generate.py ──JSONL──► qa_dataset.py ──freeze──► Frozen  │
│   ▲            ▲                                              dataset │
│ Haiku writers  └── ingest_longform.py ◄── long-form texts            │
└──────────────────────────────────────────────────────────────┬───────┘
                                                               │ train split
HF Hub (kev-4b, Qwen3.5-4B) ──init_from──► train.py (QLoRA NF4) ◄┘
                                               │ save
                                               ▼
                 Your app ──check(text)──► PDPAGuard ◄──load 4-bit── runs/pdpa-4b
                                                                       │
            OpenThai-SystemOne ──► evaluate.py ◄── kev-base ◄── HF Hub  │
                                       │ ◄──────────── kev-pdpa ───────┘
                                       ▼
                                  compare.py → results/comparison.md + charts
```

---

## 6. Every function: what it does, input → output

### 6.1 `src/pdpa_guard/guard.py`: the main API (what most people will use)

| Function / class | Does | Input | Output |
|---|---|---|---|
| `PDPAGuard(model, device="cuda", load_4bit=True, threshold=0.5)` | Loads the model (Qwen3.5-4B 4-bit backbone + LoRA adapter + pointer head) | Hub id or a run folder | A ready-to-use object |
| `PDPAGuard.check(state, lang="th")` | Checks one text | `str`, or a dict/list (JSON) | `PDPAResult` |
| `PDPAGuard.check_many(states)` | Checks several texts | `list` | `list[PDPAResult]` |
| `PDPAGuard.raw(state)` | Lowest level: the raw probabilities of every question | text | `(answers: dict, seconds: float)` |
| `canonical_questions(lang)` | Builds the 11 questions in a fixed wording | `"th"` / `"en"` | dict of questions in the TypeSafe/Kev format |
| `PDPAResult.detected(threshold)` | Lists the `has_*` flags above the threshold | float | `list[str]` |
| `main()` | Command-line interface | argv | Prints the result |

Fields of **`PDPAResult`**: `is_sensitive`, `p_sensitive`, `risk_level` (0–4), `risk_score` (expected value, e.g. 3.4), `risk_label`, `risk_probs`, `primary_category`, `category_probs`, `flags` (8 probabilities), `latency_ms`

### 6.2 `src/pdpa_guard/taxonomy.py`: the PDPA rules (single source of truth)

| Function | Does | Input | Output |
|---|---|---|---|
| `effective_spans(spans)` | Drops section 26 data with no identifiable person | list of spans `{type, value}` | Filtered spans |
| `compute_risk(spans)` | Computes the risk level from the rules in section 3 | spans | `int` 0–4 |
| `primary_category(spans)` | Picks the highest-risk category | spans | `str` (one of 10 categories) |
| `has_flags(spans)` | Computes the 8 `has_*` flags | spans | `dict[str, bool]` |
| `build_questions(spans, rng, lang)` | Builds the 11 questions + labels (randomly choosing among ≥4 wordings per question) | spans, RNG | dict of questions with `label` |
| `labels_from_spans(spans)` | Labels only (used by QA/eval) | spans | `dict` |

### 6.3 `src/pdpa_guard/fakers.py`: realistic-looking fake data generators

| Function | Does | Output example |
|---|---|---|
| `make_thai_id(rng)` / `is_valid_thai_id(s)` | Generates/checks a 13-digit national ID with a **correct mod-11 checksum** | `3-1012-04567-89-2` |
| `make_invalid_id(rng)` | 13 digits that **fail** the checksum (used as hard negatives) | Order number `6903953791880` |
| `make_card(rng)` / `luhn_ok(s)` | Generates/checks a card number that **passes Luhn** with a real prefix (Visa 4, MC 51–55, JCB 35) | `4539 1488 0343 6467` |
| `s_name`, `s_phone`, `s_email`, `s_address`, `s_dob`, `s_passport`, `s_bank_account`, `s_card_full`, `s_password`, `s_otp`, `s_health`, `s_special`, `s_plate`, `s_ip`, `s_line_id`, … | Slot functions (48 slots) for each data type, in varied formats (dashes, spaces, Thai digits) | Thai names, real Thai provinces/districts/postcodes, Thai license plates |
| `s_order_no`, company/public-figure/masked-value slots | Data that **looks sensitive but isn't** | `TK-2026-4356`, `**** 7609` |

### 6.4 `src/pdpa_guard/generate.py`: assembling the dataset

| Function | Does |
|---|---|
| `load_templates(path)` | Reads the templates (`{name}`, `{phone}` … slots) |
| `instantiate(tpl, rng)` | Fills the slots with fake values and records `pii_spans` |
| `split_dev_templates(...)` | Sets aside a separate group of templates for dev (no leakage) |
| `make_split(...)` | Generates records for each split, controlling class balance and duplicates |
| `main()` | Writes `data/processed/{train,dev,test}.jsonl` (with meta) and `data/processed/kev/*.jsonl` (Kev training format) |

### 6.5 `src/kev_qlora/`: QLoRA patch + evaluation tools

| File / function | Does | Input → Output |
|---|---|---|
| `qlora.bnb_config()` | Config for 4-bit NF4 + double quant + bf16 compute | → `BitsAndBytesConfig` |
| `qlora.load_quantized_backbone(...)` | Loads Qwen3.5-4B as 4-bit straight onto the GPU | model name → backbone |
| `qlora.prepare_kbit(lm)` | Freezes the base, enables gradient checkpointing, keeps embeddings in bf16 (saves 2.5 GB compared with peft's default) | backbone → ready to train |
| `qlora.make_class()` | Builds `QLoRADecisionModel`, a subclass of Kev's `DecisionModel` that supports 4-bit | → class |
| `qlora.install()` | Patches `kev.train` to use the 4-bit model (the original Kev code isn't modified) | → holder |
| `qlora.load_adapter_inference(run, device)` | Loads adapter + head + calibration temperature for inference | run dir → `(tokenizer, model)` |
| `systems.KevSystem.predict(state, questions)` | Runs a Kev model on one request | → `(answers, seconds)` |
| `systems.get_system(name)` | Selects a system: `kev-base`, `kev-pdpa`, `openthai` | → system |
| `openthai_adapter.OpenThaiSystem` | Runs `iapp/OpenThai-SystemOne` via transformers or Ollama with the same questions | → `(answers, seconds)` |
| `metrics_pdpa.compute_metrics(rows)` | Computes accuracy, P/R/F1, macro-F1, risk MAE, Brier, ECE, latency | prediction rows → dict of metrics |
| `compat.*` | Shims for `resource`/`fcntl`, which don't exist on Windows | — |

### 6.6 `scripts/`: commands to run

| Script | Does | Main output |
|---|---|---|
| `make_value_packs.py` | Generates PII "value packs" for the long-form writers | `data/longform/packs_*.jsonl` |
| `ingest_longform.py` | Checks that every PII value appears verbatim in the text, and drops records that don't | `data/longform/N_A.jsonl`, `N_B.jsonl` |
| `validate_templates.py` / `merge_templates.py` | Validates/merges the templates | `data/templates/*.jsonl` |
| `qa_dataset.py` | **Quality gate**: recomputes the labels, checksums, Luhn, duplicates, template leakage (exits non-zero on any error) | `QA_REPORT.md` |
| `train.py` | QLoRA training (wraps `kev.train`), logs step/loss/VRAM | `runs/pdpa-4b/` |
| `evaluate.py` | Runs one system on the test set (resumable) | `results/eval_<system>.json` |
| `compare.py` | Builds the comparison table + charts | `results/comparison.md`, `*.png` |
| `run_pipeline.sh` | train → eval of 3 systems → compare, in one go | everything above |
| `tune_optuna.py` | Hyperparameter search with Optuna (TPE + MedianPruner), **minimizing dev NLL** (not train loss, to avoid overfitting); resumable | `results/optuna/best_params.json`, `trials.md`, `optuna_history.png` |

---

## 7. Workflows between functions

### 7.1 When you call `guard.check(text)` (inference)
See the diagram [`check-sequence.html`](docs/architecture/check-sequence.html)

1. **`PDPAGuard.check`** receives the text → calls **`canonical_questions("th")`** to get the 11 questions
2. Passes them to **`KevSystem.predict(state, questions)`**
3. `predict` → **`kev.api.to_record()`** validates the request format and converts JSON into labelled text
4. → **`DecisionModel.encode` + prefill**: the Qwen3.5-4B (4-bit) + LoRA backbone reads the text **once**, and each of the 11 questions is a "row" that reuses the cache of that text
5. → the **pointer head** turns the hidden states into logits per option → divides by the temperature T → softmax = **calibrated probabilities**
6. `check` packages the result as a **`PDPAResult`** (picks the level/category with the highest probability, computes the expected risk value)

**Data passed between functions:**
`str` → `{11 questions}` → `record (state text + option keys)` → `token ids` → `hidden states [rows × 2560]` → `probs per option` → `PDPAResult`

### 7.2 Building the dataset (data pipeline)
See [`dataset-dataflow.html`](docs/architecture/dataset-dataflow.html)

```
Template path:   templates + fakers.s_*() ──► generate.instantiate() ──► pii_spans
Long-form path:  fakers ──► make_value_packs ──► Haiku writes text around the values ──► ingest_longform (verbatim check)
Both paths:      pii_spans ──► taxonomy.build_questions() (labels computed by code) ──► qa_dataset.py ──► FROZEN
```
**The key point:** labels are always computed by `taxonomy.py` from `pii_spans`. No person or model ever "picks a label" itself, so the labels are consistent across the whole dataset.

### 7.3 Training → evaluation
```
data/processed/kev/train.jsonl ──► scripts/train.py ──(qlora.install patches kev.train)──► runs/pdpa-4b/
runs/pdpa-4b ─┐
kev-4b base  ─┼──► scripts/evaluate.py (identical state + questions for every system; input_sha checks this) ──► results/eval_*.json
OpenThai     ─┘                                                                  └──► scripts/compare.py ──► comparison.md
```

---

## 8. Dataset

Published at [`Datchthana/thai-pdpa-pii-synthetic`](https://huggingface.co/datasets/Datchthana/thai-pdpa-pii-synthetic)

| Split | Records | Positive | Hard negatives | Clean negatives | Long-form texts |
|---|---|---|---|---|---|
| train | 2,741 | 55.9% | 25.1% | 19.0% | 341 |
| dev | 450 | 56.2% | 25.3% | 18.4% | 50 |
| test | 735 | 57.4% | 24.1% | 18.5% | 115 |

**Quality was prioritized over quantity:**
- **15 genres**: LINE chat, support ticket, shipping form, job application, hospital note, bank/loan, school records, government form, social post, email, server log/JSON, rental contract, insurance claim, news, product review
- **Every value is valid**: national IDs pass the mod-11 checksum, card numbers pass Luhn, provinces/districts/postcodes are real
- **Hard negatives that look sensitive but aren't**: 13-digit order numbers that fail the checksum, card numbers that fail Luhn, public figures in the news, company addresses/phone numbers (juristic persons aren't covered by the PDPA), masked data, anonymous health statistics
- **506 natural long-form texts** (300–1,200 characters) written by 11 Haiku writers, with every PII value checked to appear verbatim
- **No leakage**: test uses B templates written by **different authors** from the A templates used for train/dev; there are 0 near-duplicates across splits (5-gram Jaccard ≥ 0.8)
- Mixed Thai/English instructions (15% English), with at least 4 wordings per question so the model doesn't memorize one phrasing

Risk distribution in test: 0 = 313 · 1 = 55 · 2 = 60 · 3 = 87 · 4 = 220 (the full report is in [`data/processed/QA_REPORT.md`](data/processed/QA_REPORT.md))

---

## 9. How we trained it

| Item | Value |
|---|---|
| Starting model | `jaredpalmer/kev-4b` (LoRA r16 + pointer head on `Qwen/Qwen3.5-4B-Base`) via `--init_from` |
| Quantization | NF4 4-bit + double quant, bf16 compute (bitsandbytes); the DeltaNet gates `in_proj_a/b` stay bf16 |
| LoRA | rank 16, α 32; targets q/k/v/o, gate/up/down, and the Gated DeltaNet projections (same as Kev); 33.8M trainable params |
| Optimizer | AdamW, lr 2e-5 one-cycle, weight decay 0.01 |
| Batch | 1 × accumulation 8; `--shared_prefix 1` (each text runs once, questions share its cache); `--length_sort 1` |
| Epochs | 1 (343 optimizer steps) |
| max_state | 1,024 tokens |
| Kernels | flash-linear-attention 0.5.2 + triton-windows 3.4: about **10× faster** than the torch fallback |
| Hardware | i5-12500H, 16 GB RAM, **RTX 4060 Laptop 8 GB** (peak VRAM ≈ 4.5 GB) |
| Time | ≈ 3 hours |

Reproduce:
```bash
bash scripts/setup_env.sh
# the dataset lives on HF (not in git, because GitHub push protection blocks the fake API keys in the credential examples)
hf download Datchthana/thai-pdpa-pii-synthetic --repo-type dataset --local-dir data/processed
bash scripts/run_pipeline.sh            # train + eval + compare
```
> To regenerate the data from scratch: `python src/pdpa_guard/generate.py` (needs `data/longform/` from the long-form step) then `python scripts/qa_dataset.py`

---

## 10. Limitations

- **The data is synthetic.** It's realistic but not real-world data. Real-world texts (scanned documents, OCR, heavy slang) may score lower.
- The data is mainly Thai; English shows up only in mixed text and in 15% of instructions.
- Some long-form records have a condition that doesn't match the person (e.g. a disease specific to one sex) because the values come from random value packs. The labels are still correct.
- The test set has few `name` records as the primary category (14).
- `risk_level` follows the rules **this project defined** from PDPA principles. It isn't an official standard from the regulator (PDPC).
- The model reads only the first 1,024 tokens it was trained on. For longer documents, split them into chunks first.
- Don't use it as the sole basis for decisions with legal consequences. Send uncertain cases (p between 0.3 and 0.9) to a person for review.

---

## 11. How the team worked (agent swarm)

This project was built by a team of AI agents arranged like a project team:

| Role | Model | Count | Responsibility |
|---|---|---|---|
| Senior Project Manager | Claude **Opus** | 1 | Spec (`docs/SPEC.md`), review/quality gates, GPU scheduling, QLoRA training + optimization (fla kernels), architecture diagrams, README/Model Card, release |
| Assistant PM: data-lead | Claude **Sonnet** | 1 | Designed the taxonomy, generator, QA; supervised the Haiku devs; 2 rounds of fixes following PM review |
| Assistant PM: train-lead | Claude **Sonnet** | 1 | Environment, QLoRA patch, train wrapper, eval harness, verified metrics independently |
| Developers | Claude **Haiku** | 27 | Data round 1 (9): fakers.py, templates A1–A4 / B1–B3 (different authors for train/test). Data round 2 (16): 3 filler rewirers, A5/B4 templates, 11 long-form writers (8 for train/dev, 3 for test). Train/eval (2): evaluate/compare/metrics, OpenThai adapter |

Quality gates that were enforced: the PM rejected the first data round (unrealistic filler values, too few section-26 test records, texts too short) → round 2 added 506 long-form records and rebalanced the test set → the data was frozen with sha256 before training.

---

## License and acknowledgements
- Adapter, head, code and dataset: **Apache-2.0**
- Base model: [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) (Apache-2.0); [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer (Apache-2.0)
- Compared against [iapp/OpenThai-SystemOne](https://huggingface.co/iapp/OpenThai-SystemOne) (Apache-2.0)
- Diagrams by [archify](https://github.com/tt-a1i/archify) (MIT)
