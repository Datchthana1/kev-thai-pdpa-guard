"""Jev (TypeSafe's hosted System One model, `typesafe-ai/jev`) as an evaluation system, through the Vercel AI Gateway.

Needs AI_GATEWAY_API_KEY in the environment (or in a git-ignored `.env` at the project root) and the Node worker in
tools/jev_worker (`npm install` there). Each request is sent with zero data retention requested; every call is
counted against a USD budget at Jev's list price, and the run stops before the budget would be exceeded.

    answers, latency_s = JevSystem(budget_usd=1.0).predict(state, questions)
"""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tools" / "jev_worker" / "jev-evaluate.mjs"
PRICE_PER_MILLION = 0.042   # Jev list price per million input tokens (same constant as kev.predictors)


def _load_dotenv():
    env = ROOT / ".env"
    if env.exists() and not os.environ.get("AI_GATEWAY_API_KEY"):
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("AI_GATEWAY_API_KEY="):
                os.environ["AI_GATEWAY_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


class JevSystem:
    name = "typesafe-ai/jev"

    def __init__(self, budget_usd=1.0, attempts=4, **_):
        _load_dotenv()
        if not os.environ.get("AI_GATEWAY_API_KEY"):
            raise RuntimeError("Set AI_GATEWAY_API_KEY (environment or .env) to evaluate Jev")
        self.budget, self.attempts = budget_usd, attempts
        self.calls = self.input_tokens = self.output_tokens = 0
        self.proc = subprocess.Popen(["node", str(WORKER)], cwd=WORKER.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="utf-8", env=dict(os.environ))

    def predict(self, state, questions):
        if (self.input_tokens + 65536) * PRICE_PER_MILLION / 1e6 > self.budget:
            raise RuntimeError(f"Jev budget cap reached (${self.budget:.2f})")
        req = {"state": state, "questions": questions}
        for attempt in range(self.attempts):
            t0 = time.perf_counter()
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            latency = time.perf_counter() - t0
            if not line:
                raise RuntimeError("Jev worker exited without a response")
            res = json.loads(line)
            self.calls += 1
            if "error" not in res:
                break
            status = res["error"].get("status")
            if attempt == self.attempts - 1 or (status is not None and status < 500):
                raise RuntimeError(f"Jev request failed: {res['error'].get('name')} (HTTP {status})")
            time.sleep(min(2 ** attempt, 30))
        usage = res.get("usage") or {}
        self.input_tokens += usage.get("inputTokens") or 0
        self.output_tokens += usage.get("outputTokens") or 0
        return {qid: self._answer(q, res["answers"][qid]) for qid, q in questions.items()}, latency

    @staticmethod
    def _answer(q, a):
        if q["type"] == "noul":
            p = float(a["probability"])
            return {"type": "noul", "probs": {"false": 1 - p, "true": p}}
        probs = a["probabilities"]
        if q["type"] == "score":
            crit = q.get("criteria") or []
            keys = list(probs)
            if not all(str(k).isdigit() for k in keys):   # levels keyed by their description -> index
                probs = {str(crit.index(k)): v for k, v in probs.items() if k in crit}
            return {"type": "score", "probs": {str(k): float(v) for k, v in probs.items()}}
        return {"type": "choice", "probs": {k: float(v) for k, v in probs.items()}}

    def accounting(self):
        return {"model": self.name, "calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "estimated_usd": round(self.input_tokens * PRICE_PER_MILLION / 1e6, 4), "budget_usd": self.budget,
                "sdk": "ai@7.0.105 via Vercel AI Gateway", "zero_data_retention_requested": True}

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
