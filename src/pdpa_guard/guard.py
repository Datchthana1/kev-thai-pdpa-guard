"""High-level API: check a Thai/English text for PDPA personal data with the fine-tuned Kev-4B model.

    from pdpa_guard.guard import PDPAGuard
    guard = PDPAGuard("Datchthana/kev-thai-pdpa-guard")       # or a local run dir such as runs/pdpa-4b
    result = guard.check("โอนเงินให้คุณสมชาย ใจดี เลขบัญชี 123-4-56789-0 กสิกรไทย")
    print(result.is_sensitive, result.risk_level, result.risk_label, result.primary_category)

CLI:
    python -m pdpa_guard.guard "ข้อความที่ต้องการตรวจ" [--model runs/pdpa-4b] [--lang th|en] [--json]

All 11 questions are asked in one call; the model scores them in a single prefill pass over the text
(no text generation), so every answer is a calibrated probability.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field

from .taxonomy import HAS_QUESTIONS, build_questions

DEFAULT_MODEL = "Datchthana/kev-thai-pdpa-guard"
RISK_LABELS = ["none", "low", "medium", "high", "critical"]
RISK_LABELS_TH = ["ไม่มี", "ต่ำ", "ปานกลาง", "สูง", "วิกฤต"]


class _First:
    """A stand-in RNG whose choice() always returns the first paraphrase, so the questions are deterministic."""

    def choice(self, seq):
        return seq[0]

    def random(self):
        return 1.0


def canonical_questions(lang: str = "th") -> dict:
    """The 11 questions exactly as asked at inference (labels stripped)."""
    qs = build_questions([], _First(), lang=lang)
    return {qid: {k: v for k, v in q.items() if k != "label"} for qid, q in qs.items()}


@dataclass
class PDPAResult:
    is_sensitive: bool                 # p_sensitive >= threshold
    p_sensitive: float                 # probability the text holds personal data of an identifiable person
    risk_level: int                    # 0..4, the most likely level
    risk_score: float                  # expected level (probability-weighted), e.g. 3.4
    risk_label: str                    # none / low / medium / high / critical
    risk_probs: dict                   # {"0": p, ..., "4": p}
    primary_category: str              # highest-risk category present (see taxonomy.CATEGORIES)
    category_probs: dict               # {category: p}
    flags: dict = field(default_factory=dict)   # {"has_name": p_yes, ...} for the 8 per-type questions
    latency_ms: float = 0.0

    def detected(self, threshold: float = 0.5) -> list[str]:
        """Names of the per-type flags whose probability is at least `threshold`."""
        return [k for k, p in self.flags.items() if p >= threshold]

    def to_dict(self) -> dict:
        return asdict(self)


class PDPAGuard:
    """Wraps a Kev decision model (run dir or Hub id) and answers the 11 PDPA questions for a text.

    model:     Hub id or local run dir holding adapter_model.safetensors + head.pt
    device:    "cuda" (default) or "cpu"
    load_4bit: keep the Qwen3.5-4B backbone in NF4 so it fits an 8 GB GPU (default True)
    threshold: probability above which is_sensitive is True
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cuda", load_4bit: bool = True, threshold: float = 0.5):
        from kev_qlora.systems import KevSystem
        self.system = KevSystem(model, device=device, load_4bit=load_4bit)
        self.threshold = threshold

    def raw(self, state, lang: str = "th") -> tuple[dict, float]:
        """Lowest level: send `state` (text, dict or list) with the canonical questions; returns (answers, seconds)."""
        return self.system.predict(state, canonical_questions(lang))

    def check(self, state, lang: str = "th") -> PDPAResult:
        """Score one text (or JSON object) and return a PDPAResult."""
        ans, dt = self.raw(state, lang)
        p_sens = ans["is_sensitive"]["probs"]["true"]
        rp = ans["risk_level"]["probs"]
        level = int(max(rp, key=rp.get))
        cp = ans["primary_category"]["probs"]
        return PDPAResult(
            is_sensitive=p_sens >= self.threshold,
            p_sensitive=round(p_sens, 4),
            risk_level=level,
            risk_score=round(sum(int(k) * v for k, v in rp.items()), 3),
            risk_label=RISK_LABELS[level],
            risk_probs={k: round(v, 4) for k, v in rp.items()},
            primary_category=max(cp, key=cp.get),
            category_probs={k: round(v, 4) for k, v in sorted(cp.items(), key=lambda kv: -kv[1])},
            flags={q: round(ans[q]["probs"]["true"], 4) for q in HAS_QUESTIONS},
            latency_ms=round(dt * 1000, 1),
        )

    def check_many(self, states, lang: str = "th") -> list[PDPAResult]:
        """Score several texts one after another."""
        return [self.check(s, lang) for s in states]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check a text for PDPA personal data (Kev-4B fine-tuned).")
    ap.add_argument("text", nargs="?", help="text to check; read from stdin when omitted")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--lang", default="th", choices=["th", "en"])
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args(argv)
    text = a.text if a.text is not None else sys.stdin.read()
    r = PDPAGuard(a.model, device=a.device).check(text, lang=a.lang)
    if a.json:
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=1))
    else:
        print(f"sensitive : {r.is_sensitive} (p={r.p_sensitive:.3f})")
        print(f"risk      : {r.risk_level} {r.risk_label} / {RISK_LABELS_TH[r.risk_level]} (expected {r.risk_score:.2f})")
        print(f"category  : {r.primary_category} ({r.category_probs[r.primary_category]:.3f})")
        print(f"detected  : {', '.join(r.detected()) or '-'}")
        print(f"latency   : {r.latency_ms} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
