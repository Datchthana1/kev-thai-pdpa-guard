"""System registry for scripts/evaluate.py.

    system = get_system("kev-base" | "kev-pdpa" | "openthai", run=..., device="cuda", load_4bit=True)
    answers, latency_s = system.predict(state, questions)

`questions` is the TypeSafe/Kev request shape ({qid: {"type", "instructions", "criteria"?}}), identical for every system.
`answers` = {qid: {"type": ..., "probs": {key: p}}}; keys "false"/"true" (noul), "0".."N-1" (score), option names (choice).
"""
import time

KEV_BASE_ID = "jaredpalmer/kev-4b"


class KevSystem:
    """In-process Kev scorer (torch backend). load_4bit=True keeps the backbone NF4 (fits 8 GB), adapter unmerged."""

    def __init__(self, run, device="cuda", load_4bit=True, max_state=2048, max_branch=1024):
        from . import compat
        compat.set_env_defaults()
        compat.install_resource_shim()
        compat.install_fcntl_shim()
        import torch
        from . import qlora
        self.torch = torch
        self.device = device
        self.run = run
        self.max_state, self.max_branch = max_state, max_branch
        self.tok, self.model = qlora.load_adapter_inference(run, device, load_4bit=load_4bit,
                                                            dtype=torch.bfloat16 if device != "cpu" else torch.float32)
        self.name = run

    def predict(self, state, questions):
        from kev.api import SystemOneRequest, to_record
        torch = self.torch
        rec, meta = to_record(SystemOneRequest.model_validate({"state": state, "questions": questions}))
        enc = self.model.encode(self.tok, rec, max_state=self.max_state, max_branch=self.max_branch, strict=False)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            ps = self.model.probs(enc)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        answers = {m["id"]: {"type": m["type"], "probs": dict(zip(m["keys"], [float(x) for x in p.tolist()]))} for m, p in zip(meta, ps)}
        return answers, dt


def get_system(name, run=None, device="cuda", load_4bit=True, **kw):
    if name == "kev-base":
        return KevSystem(run or KEV_BASE_ID, device=device, load_4bit=load_4bit, **kw)
    if name == "kev-pdpa":
        if not run:
            raise ValueError("kev-pdpa needs --run <run dir>")
        return KevSystem(run, device=device, load_4bit=load_4bit, **kw)
    if name == "openthai":
        from .openthai_adapter import OpenThaiSystem
        return OpenThaiSystem(backend="transformers", device=device, **kw)
    if name == "tev1":
        # togethercomputer/Tev1-4B-experimental (4.3B, Q8_0) through Ollama's /v1/systemone, which turns the model's
        # option-letter probabilities into choice / noul / score answers. Ollama decides GPU placement.
        from .openthai_adapter import OpenThaiSystem
        return OpenThaiSystem(backend="ollama", ollama_model="tev1:4b", **kw)
    raise ValueError(f"unknown system {name!r}")
