# PDPA guard evaluation: comparison

Systems: `kev-base`, `kev-pdpa`, `openthai`, `tev1`

Records per system: kev-base=735, kev-pdpa=735, openthai=735, tev1=735. Bold = best per row (higher is better for F1/accuracy/precision/recall; lower for MAE, Brier, ECE, latency). Brier is the multi-class sum over classes, averaged over questions; ECE uses 15 equal-width bins.

## Headline metrics

| metric | kev-base | kev-pdpa | openthai | tev1 |
|---|---|---|---|---|
| n_records | 735 | 735 | 735 | 735 |
| is_sensitive precision | 0.876 | **0.993** | 0.968 | 0.866 |
| is_sensitive recall | 0.905 | **0.988** | 0.287 | 0.945 |
| is_sensitive F1 | 0.890 | **0.990** | 0.442 | 0.904 |
| has_* macro-F1 (8 nouls) | 0.649 | **0.934** | 0.620 | 0.671 |
| primary_category accuracy | 0.638 | **0.946** | 0.463 | 0.559 |
| primary_category macro-F1 | 0.565 | **0.931** | 0.436 | 0.520 |
| risk_level exact accuracy | 0.476 | **0.912** | 0.524 | 0.486 |
| risk_level +-1 accuracy | 0.703 | **0.977** | 0.702 | 0.702 |
| risk_level MAE (argmax) | 1.033 | **0.125** | 1.083 | 0.913 |
| risk_level MAE (expected) | 1.214 | **0.139** | 1.121 | 0.947 |
| Brier (all questions) | 0.247 | **0.040** | 0.295 | 0.253 |
| ECE (all questions, 15 bins) | 0.032 | **0.011** | 0.063 | 0.039 |
| latency mean (s) | 1.309 | 1.275 | **0.116** | 3.267 |
| latency p50 (s) | 1.316 | 1.277 | **0.116** | 3.265 |
| latency p95 (s) | 1.395 | 1.368 | **0.161** | 3.400 |

## Per-question accuracy (every qid)

| qid | kev-base | kev-pdpa | openthai | tev1 |
|---|---|---|---|---|
| is_sensitive | 0.872 | **0.989** | 0.585 | 0.884 |
| risk_level | 0.476 | **0.912** | 0.524 | 0.486 |
| primary_category | 0.638 | **0.946** | 0.463 | 0.559 |
| has_name | 0.959 | **0.969** | 0.824 | 0.963 |
| has_contact | 0.927 | **0.992** | 0.917 | 0.950 |
| has_address | 0.941 | **0.996** | 0.944 | 0.944 |
| has_national_id | 0.966 | **0.997** | 0.946 | 0.973 |
| has_financial | 0.863 | **0.993** | 0.899 | 0.871 |
| has_credential | 0.811 | **0.992** | 0.898 | 0.906 |
| has_health_or_special | 0.718 | **0.985** | 0.884 | 0.736 |
| has_other_identifier | 0.824 | **0.947** | 0.878 | 0.800 |

## By genre

### is_sensitive F1

| genre | kev-base | kev-pdpa | openthai | tev1 |
|---|---|---|---|---|
| bank_loan (n=39) | 0.864 | **0.974** | 0.593 | 0.864 |
| ecommerce_delivery (n=51) | 0.844 | **0.984** | 0.488 | 0.829 |
| email (n=58) | 0.935 | **1.000** | 0.458 | 0.949 |
| gov_form (n=49) | 0.963 | **1.000** | 0.600 | 0.962 |
| hospital_note (n=37) | 0.895 | **1.000** | 0.615 | 0.872 |
| hr_application (n=47) | 0.981 | **1.000** | 0.312 | 0.981 |
| insurance_claim (n=43) | 0.941 | **1.000** | 0.387 | 0.941 |
| line_chat (n=51) | 0.833 | **0.984** | 0.410 | 0.862 |
| news_article (n=47) | 0.857 | **1.000** | 0.429 | 0.889 |
| realestate_contract (n=50) | 0.945 | **1.000** | 0.444 | 0.982 |
| review (n=45) | 0.816 | **0.978** | 0.483 | 0.913 |
| school_record (n=50) | 0.912 | **1.000** | 0.424 | 0.929 |
| server_log (n=51) | 0.951 | **0.984** | 0.400 | 0.939 |
| social_post (n=65) | 0.795 | **0.977** | 0.386 | 0.841 |
| support_ticket (n=52) | 0.862 | **0.986** | 0.300 | 0.849 |

### risk_level exact accuracy

| genre | kev-base | kev-pdpa | openthai | tev1 |
|---|---|---|---|---|
| bank_loan (n=39) | 0.410 | **0.846** | 0.513 | 0.436 |
| ecommerce_delivery (n=51) | 0.451 | **0.922** | 0.510 | 0.314 |
| email (n=58) | 0.362 | **0.948** | 0.552 | 0.379 |
| gov_form (n=49) | 0.531 | **0.939** | 0.571 | 0.633 |
| hospital_note (n=37) | 0.378 | **0.973** | 0.486 | 0.568 |
| hr_application (n=47) | 0.489 | **0.957** | 0.468 | 0.574 |
| insurance_claim (n=43) | 0.442 | **0.837** | 0.512 | 0.488 |
| line_chat (n=51) | 0.490 | **0.902** | 0.569 | 0.412 |
| news_article (n=47) | 0.574 | **0.936** | 0.660 | 0.638 |
| realestate_contract (n=50) | 0.620 | **0.860** | 0.580 | 0.600 |
| review (n=45) | 0.733 | **0.978** | 0.689 | 0.689 |
| school_record (n=50) | 0.380 | **0.940** | 0.440 | 0.380 |
| server_log (n=51) | 0.490 | **0.902** | 0.529 | 0.490 |
| social_post (n=65) | 0.462 | **0.877** | 0.385 | 0.508 |
| support_ticket (n=52) | 0.346 | **0.865** | 0.442 | 0.250 |

### Hard negatives vs not

| slice | kev-base | kev-pdpa | openthai | tev1 |
|---|---|---|---|---|
| hard_negative=true: is_sensitive F1 | **0.000** | **0.000** | **0.000** | **0.000** |
| hard_negative=true: risk exact acc | 0.503 | **0.966** | 0.746 | 0.480 |
| hard_negative=false: is_sensitive F1 | 0.948 | **0.994** | 0.445 | 0.968 |
| hard_negative=false: risk exact acc | 0.468 | **0.894** | 0.453 | 0.487 |

## By gold risk level

| gold risk | kev-base | kev-pdpa | openthai | tev1 |
|---|---|---|---|---|
| 0 | acc 0.709 / MAE 0.757 (n=313) | acc 0.981 / MAE 0.058 (n=313) | acc 0.840 / MAE 0.546 (n=313) | acc 0.668 / MAE 0.837 (n=313) |
| 1 | acc 0.327 / MAE 0.745 (n=55) | acc 0.927 / MAE 0.145 (n=55) | acc 0.200 / MAE 0.800 (n=55) | acc 0.273 / MAE 0.727 (n=55) |
| 2 | acc 0.983 / MAE 0.017 (n=60) | acc 0.917 / MAE 0.100 (n=60) | acc 0.650 / MAE 0.683 (n=60) | acc 1.000 / MAE 0.000 (n=60) |
| 3 | acc 0.195 / MAE 0.805 (n=87) | acc 0.667 / MAE 0.391 (n=87) | acc 0.034 / MAE 1.322 (n=87) | acc 0.195 / MAE 0.805 (n=87) |
| 4 | acc 0.155 / MAE 1.864 (n=220) | acc 0.905 / MAE 0.118 (n=220) | acc 0.314 / MAE 1.932 (n=220) | acc 0.255 / MAE 1.359 (n=220) |

## Calibration by question type

| type / metric | kev-base | kev-pdpa | openthai | tev1 |
|---|---|---|---|---|
| noul Brier | 0.173 | **0.024** | 0.201 | 0.151 |
| noul ECE | 0.041 | **0.006** | 0.030 | 0.019 |
| score Brier | 0.668 | **0.135** | 0.694 | 0.788 |
| score ECE | 0.121 | **0.043** | 0.230 | 0.319 |
| choice Brier | 0.492 | **0.092** | 0.739 | 0.644 |
| choice ECE | 0.044 | **0.036** | 0.193 | 0.231 |
| overall Brier | 0.247 | **0.040** | 0.295 | 0.253 |
| overall ECE | 0.032 | **0.011** | 0.063 | 0.039 |

Brier and ECE per question are in `results/eval_<system>.json` under `metrics.calibration.by_qid`.
