"""Adapter for OpenThai-SystemOne (iapp/OpenThai-SystemOne), a 0.8B Thai/English "System One" decision model.

Two interchangeable backends behind one call, ``OpenThaiSystem.predict(state, questions)``:

``transformers``
    Loads ``iapp/OpenThai-SystemOne`` with ``trust_remote_code=True`` (the repo ships its own
    ``modeling.py`` / ``formatting.py`` / ``types.py``). The remote ``Formatter`` builds the token sequence and the
    remote ``OpenThaiSystemOneForDecision`` returns a 256-slot probability vector per question from one forward pass.
    Decoding mirrors the official ``openthai_systemone.client.SystemOneClient._decode``: the probabilities of the real
    options are renormalised (slot 255 = "abstain" is reported separately as ``abstain``), noul = p(slot 1 = yes),
    score = distribution over levels 0..N-1 with expected level as ``score``. Default dtype is float32 on CPU
    (bf16 is supported via ``dtype="bfloat16"``). Only ``device`` decides where this runs.

``ollama``
    POSTs to ``{ollama_url}/v1/systemone`` (Ollama >= 0.35 with the ``iapp/openthai-systemone`` build). Ollama itself
    decides device placement (CPU/GPU) - ``device`` is ignored by this backend. The Ollama build returns answers as
    points/labels rather than full distributions for some types, so they are converted as follows:

    * noul:   ``{"noul": p}``                 -> ``{"true": p, "false": 1 - p}``.
    * choice: ``"probabilities"`` used if present (renormalised). Otherwise ``"choice"`` + ``"confidence"`` c:
              the label gets mass c and the remaining options share (1 - c) equally (one-hot if no confidence).
              Note: ``confidence`` in this API is 1 - normalised entropy, so this is an approximation of the
              distribution, not the model's own distribution; use ``probabilities`` whenever the server gives it.
    * score:  ``"probabilities"`` used if present. Otherwise a scalar expected level ``s`` is spread with a
              triangular kernel of half-width 1 over the integer levels: ``p(i) = max(0, 1 - |i - s|)``, normalised
              after clamping ``s`` to ``[0, N-1]``. This keeps the expected level equal to ``s`` for interior values
              (for ``s`` in range the kernel's mean is exactly ``s``), but its spread is a modelling choice.

Output contract (both backends), with ``questions`` as the Kev/TypeSafe request shape::

    questions = {qid: {"type": "noul"|"score"|"choice", "instructions": str,
                       "criteria": list (score) | dict (choice) | absent/dict (noul)}}
    answers   = {qid: {"type": ..., "probs": {key: float}, ...extras}}

    keys: noul -> "false", "true";  score -> "0".."N-1" (strings);  choice -> option names.
    probs always sums to 1 (within float error).
    Extras: noul -> "noul" (p(true));  score -> "score" (expected level), "confidence";
            choice -> "choice" (argmax option), "confidence", "abstain" (transformers backend only; 0.0 for ollama
            when the server does not report it).

``predict`` returns ``(answers, latency_s)``. ``latency_s`` is wall-clock time around the model call only:
for transformers, tokenisation + forward pass (not answer decoding); for ollama, the HTTP round trip.

Environment: set ``HF_HOME=D:/hf_cache`` before first use so the weights (~1.5 GB safetensors) are not written to C:.
Nothing here touches CUDA when ``device="cpu"`` (torch may still report CUDA as available; it is not used).
"""
from __future__ import annotations

import importlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

DEFAULT_MODEL_ID = "iapp/OpenThai-SystemOne"
DEFAULT_OLLAMA_MODEL = "iapp/openthai-systemone:0.8b-q8_0"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
ABSTAIN_SLOT = 255
N_SLOTS = 256
_QTYPES = ("noul", "score", "choice")


def _check_questions(questions: Dict[str, Dict[str, Any]]) -> None:
    if not questions:
        raise ValueError("questions must be non-empty")
    for qid, q in questions.items():
        t = q.get("type")
        if t not in _QTYPES:
            raise ValueError(f"question {qid!r}: unknown type {t!r}")
        if not isinstance(q.get("instructions"), str):
            raise ValueError(f"question {qid!r}: 'instructions' must be a string")
        if t == "choice" and not isinstance(q.get("criteria"), dict):
            raise ValueError(f"question {qid!r}: choice needs 'criteria' as a dict")
        if t == "score" and not isinstance(q.get("criteria"), list):
            raise ValueError(f"question {qid!r}: score needs 'criteria' as a list")


def _normalise(d: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(v, 0.0) for v in d.values()))
    if s <= 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: max(v, 0.0) / s for k, v in d.items()}


def _entropy_confidence(probs: list) -> float:
    """1 - normalised entropy (same definition as the official client's confidence_from_probs)."""
    k = len(probs)
    if k <= 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return float(max(0.0, min(1.0, 1.0 - h / math.log(k))))


class OpenThaiSystem:
    """Run OpenThai-SystemOne through transformers (CPU) or a local Ollama server."""

    def __init__(
        self,
        backend: str = "auto",
        device: str = "cpu",
        model_id: str = DEFAULT_MODEL_ID,
        *,
        dtype: Optional[str] = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        ollama_timeout: float = 120.0,
    ):
        if backend not in ("auto", "transformers", "ollama"):
            raise ValueError("backend must be 'auto', 'transformers' or 'ollama'")
        self.backend_request = backend
        self.device = device
        self.model_id = model_id
        self.dtype = dtype  # None -> float32 on cpu, bfloat16 otherwise
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.ollama_timeout = ollama_timeout
        self.backend: Optional[str] = None
        self._tf: Optional[dict] = None  # transformers components, loaded lazily

    # ------------------------------------------------------------------ public
    def predict(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], float]:
        _check_questions(questions)
        backend = self._resolve_backend()
        if backend == "transformers":
            return self._predict_transformers(state, questions)
        return self._predict_ollama(state, questions)

    # ------------------------------------------------------------------ backend selection
    def _resolve_backend(self) -> str:
        if self.backend is not None:
            return self.backend
        if self.backend_request == "auto":
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401

                self.backend = "transformers"
            except ImportError:
                self.backend = "ollama"
        else:
            self.backend = self.backend_request
        return self.backend

    # ------------------------------------------------------------------ transformers
    def _load_transformers(self) -> dict:
        if self._tf is not None:
            return self._tf
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = self.dtype or ("float32" if self.device == "cpu" else "bfloat16")
        torch_dtype = getattr(torch, dtype)
        tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, dtype=torch_dtype)
        model = model.to(self.device).eval()

        mod = sys.modules[type(model).__module__]  # the repo's modeling.py as loaded by transformers
        if self.device != "cuda" and hasattr(mod, "use_reference_kernels"):
            mod.use_reference_kernels()  # pure-PyTorch Gated-DeltaNet paths (no triton / fla kernels on CPU)
        pkg = type(model).__module__.rsplit(".", 1)[0]
        fmt_mod = importlib.import_module(pkg + ".formatting")
        types_mod = importlib.import_module(pkg + ".types")

        fmt = fmt_mod.Formatter(tok)
        pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        self._tf = {
            "torch": torch,
            "model": model,
            "fmt": fmt,
            "fmt_mod": fmt_mod,
            "types_mod": types_mod,
            "qtype_index": getattr(mod, "QTYPE_INDEX", {"choice": 0, "score": 1, "noul": 2}),
            "pad_id": pad,
        }
        return self._tf

    def _predict_transformers(self, state, questions):
        c = self._load_transformers()
        torch, model, fmt, fmt_mod, types_mod = c["torch"], c["model"], c["fmt"], c["fmt_mod"], c["types_mod"]

        typed = {qid: types_mod.parse_question(q) for qid, q in questions.items()}
        t0 = time.perf_counter()
        enc = fmt.encode(state, typed)
        batch = fmt_mod.collate([enc], c["pad_id"])
        qtypes = torch.full_like(batch["option_counts"], -1)
        for qi, spec in enumerate(enc.specs):
            qtypes[0, qi] = c["qtype_index"][spec.qtype]
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                answer_positions=batch["answer_positions"].to(self.device),
                option_counts=batch["option_counts"].to(self.device),
                qtypes=qtypes.to(self.device),
                include_abstain=True,
            )
        probs = out.probs[0].float().cpu()
        latency = time.perf_counter() - t0

        answers: Dict[str, Dict[str, Any]] = {}
        for qi, spec in enumerate(enc.specs):
            p = probs[qi]
            k = len(spec.option_names)
            abstain = float(p[ABSTAIN_SLOT])
            pk = p[:k].tolist()
            s = sum(pk) or 1.0
            pk = [x / s for x in pk]  # renormalise over the real options (official client behaviour)
            if spec.qtype == "noul":
                p_true = pk[1]
                answers[spec.qid] = {
                    "type": "noul",
                    "probs": {"false": 1.0 - p_true, "true": p_true},
                    "noul": p_true,
                }
            elif spec.qtype == "choice":
                probs_map = {name: pk[i] for i, name in enumerate(spec.option_names)}
                best = max(probs_map, key=probs_map.get)
                answers[spec.qid] = {
                    "type": "choice",
                    "probs": probs_map,
                    "choice": best,
                    "confidence": _entropy_confidence(pk),
                    "abstain": abstain,
                }
            else:
                expected = sum(i * x for i, x in enumerate(pk))
                answers[spec.qid] = {
                    "type": "score",
                    "probs": {str(i): x for i, x in enumerate(pk)},
                    "score": expected,
                    "confidence": _entropy_confidence(pk),
                }
        return answers, latency

    # ------------------------------------------------------------------ ollama
    def _predict_ollama(self, state, questions):
        body = json.dumps(
            {"model": self.ollama_model, "state": state, "questions": questions}, ensure_ascii=False
        ).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_url + "/v1/systemone", data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.ollama_timeout) as r:
                raw = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ollama /v1/systemone failed: HTTP {e.code}: {e.read()[:500]!r}") from e
        latency = time.perf_counter() - t0
        return parse_ollama_response(raw, questions), latency


def parse_ollama_response(raw: Dict[str, Any], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert an Ollama ``/v1/systemone`` JSON body into the adapter's answer contract (see module docstring)."""
    raw_answers = raw.get("answers")
    if not isinstance(raw_answers, dict):
        raise ValueError("ollama response has no 'answers' object")
    out: Dict[str, Dict[str, Any]] = {}
    for qid, q in questions.items():
        if qid not in raw_answers:
            raise ValueError(f"ollama response is missing answer for {qid!r}")
        a = raw_answers[qid]
        t = q["type"]
        if t == "noul":
            p = float(a["noul"])
            p = min(max(p, 0.0), 1.0)
            out[qid] = {"type": "noul", "probs": {"false": 1.0 - p, "true": p}, "noul": p}
        elif t == "choice":
            names = list(q["criteria"].keys())
            if a.get("probabilities"):
                probs = _normalise({str(n): float(a["probabilities"].get(n, 0.0)) for n in names})
            else:
                label = str(a["choice"])
                conf = a.get("confidence")
                if conf is None or len(names) == 1:
                    c = 1.0
                else:
                    c = min(max(float(conf), 0.0), 1.0)
                others = [n for n in names if n != label]
                probs = {n: 0.0 for n in names}
                probs[label] = c if others else 1.0
                for n in others:
                    probs[n] = (1.0 - c) / len(others)
                probs = _normalise(probs)
            top = max(probs, key=probs.get)
            out[qid] = {
                "type": "choice",
                "probs": probs,
                "choice": str(a.get("choice", top)),
                "confidence": float(a.get("confidence") or 0.0),
                "abstain": 0.0,
            }
        else:  # score
            n_levels = len(q["criteria"])
            if a.get("probabilities"):
                probs_list = [float(a["probabilities"].get(str(i), 0.0)) for i in range(n_levels)]
                probs_list = list(_normalise({str(i): v for i, v in enumerate(probs_list)}).values())
                expected = sum(i * x for i, x in enumerate(probs_list))
            else:
                s = min(max(float(a["score"]), 0.0), float(n_levels - 1))
                tri = [max(0.0, 1.0 - abs(i - s)) for i in range(n_levels)]
                probs_list = list(_normalise({str(i): v for i, v in enumerate(tri)}).values())
                expected = s
            out[qid] = {
                "type": "score",
                "probs": {str(i): x for i, x in enumerate(probs_list)},
                "score": float(a.get("score", expected)),
                "confidence": float(a.get("confidence") or 0.0),
            }
    return out
