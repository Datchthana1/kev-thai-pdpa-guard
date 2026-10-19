"""Random-probability stand-in for kev_qlora.systems (used with EVAL_FAKE=1 and in tests).

Deterministic per (system name, state), so repeated runs give identical output.
"""

from __future__ import annotations

import random
import zlib


class FakeSystem:
    def __init__(self, name: str):
        self.name = name

    def predict(self, state, questions):
        seed = zlib.crc32(f"{self.name}|{state}".encode("utf-8"))
        rng = random.Random(seed)
        answers = {}
        for qid, q in questions.items():
            qtype = q["type"]
            if qtype == "noul":
                p = rng.random()
                probs = {"false": 1.0 - p, "true": p}
            elif qtype == "score":
                keys = [str(i) for i in range(len(q["criteria"]))]
                w = [rng.random() + 1e-6 for _ in keys]
                s = sum(w)
                probs = {k: x / s for k, x in zip(keys, w)}
            elif qtype == "choice":
                keys = list(q["criteria"].keys())
                w = [rng.random() + 1e-6 for _ in keys]
                s = sum(w)
                probs = {k: x / s for k, x in zip(keys, w)}
            else:
                raise ValueError(f"unknown question type {qtype!r}")
            answers[qid] = {"type": qtype, "probs": probs}
        latency = 0.001 + 0.004 * rng.random()
        return answers, latency


def get_system(name, run=None, device="cpu", load_4bit=False, **kw):
    return FakeSystem(name)
