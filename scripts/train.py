"""QLoRA training CLI around the official Kev trainer (kev.train.main), patched by kev_qlora.

    python scripts/train.py --data data/processed/kev/train.jsonl --dev data/processed/kev/dev.jsonl \
        --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b --load_4bit 1 --epochs 2 --lr 2e-5 \
        --batch 1 --accum 8 --max_state 1024 --out runs/pdpa-4b

Per optimizer step it prints   `step=N/T loss=.. lr=.. step_s=.. vram_gb=.. rows_s=..`   and appends the same as JSON to
<out>/train_log.jsonl. After training (checkpoint saved) it scores --dev (loss + accuracy per question type) and appends a
`{"event": "dev", ...}` line. Unknown flags are passed straight to kev.train (see `python -m kev.train -h`).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", "D:/hf_cache")

from kev_qlora import qlora  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dev", default="")
    ap.add_argument("--dev_limit", type=int, default=0, help="score only the first N dev records (0 = all)")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--init_from", default="")
    ap.add_argument("--load_4bit", type=int, choices=[0, 1], default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_state", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="", choices=["", "fp32", "bf16"], help="default: bf16 on cuda, fp32 on cpu")
    ap.add_argument("--checkpointing", type=int, choices=[0, 1], default=1)
    ap.add_argument("--out", default="runs/pdpa-4b")
    a, rest = ap.parse_known_args()
    return a, rest


def main():
    a, rest = parse()
    holder = qlora.install(bool(a.load_4bit))
    import torch
    import kev.train as kt

    dtype = a.dtype or ("bf16" if a.device in ("cuda", "auto") else "fp32")
    argv = ["kev.train", "--data", a.data, "--base", a.base, "--epochs", str(a.epochs), "--lr", str(a.lr),
            "--batch", str(a.batch), "--accum", str(a.accum), "--max_state", str(a.max_state), "--max_steps", str(a.max_steps),
            "--device", a.device, "--dtype", dtype, "--checkpointing", str(a.checkpointing), "--out", a.out]
    if a.init_from:
        argv += ["--init_from", a.init_from]
    sys.argv = argv + rest

    out = Path(a.out)
    state = {"t": time.time(), "loss": 0.0, "n": 0, "rows": 0, "step": 0, "log": None, "total": None}

    # ---- logging hooks -------------------------------------------------------------------------------------------
    orig_loss = kt.batch_loss

    def batch_loss(model, args, batch, *r, **k):
        loss, terms = orig_loss(model, args, batch, *r, **k)
        state["loss"] += float(terms["ce"]); state["n"] += sum(v.share for v in batch)
        state["rows"] += sum(len(v.rec["questions"]) for v in batch)
        return loss, terms

    kt.batch_loss = batch_loss
    Sched = torch.optim.lr_scheduler.OneCycleLR
    orig_step = Sched.step

    def sched_step(self, *args, **kw):
        if not hasattr(self, "_last_lr"):   # the scheduler's own initial step inside __init__
            return orig_step(self, *args, **kw)
        lr = self.get_last_lr()[0]
        r = orig_step(self, *args, **kw)
        now = time.time(); dt = now - state["t"]; state["t"] = now; state["step"] += 1
        vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() and a.device != "cpu" else 0.0
        rec = {"step": state["step"], "total": self.total_steps, "loss": round(state["loss"] / max(state["n"], 1), 5), "lr": lr,
               "step_s": round(dt, 2), "rows_s": round(state["rows"] / max(dt, 1e-9), 2), "vram_gb": round(vram, 2)}
        print(f"step={rec['step']}/{rec['total']} loss={rec['loss']:.4f} lr={lr:.3e} step_s={rec['step_s']} "
              f"rows_s={rec['rows_s']} vram_gb={rec['vram_gb']}", flush=True)
        if out.exists():
            with (out / "train_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        state.update(loss=0.0, n=0, rows=0)
        return r

    Sched.step = sched_step

    try:
        kt.main()
    except torch.cuda.OutOfMemoryError:
        print("OOM: CUDA out of memory (free the GPU or lower --max_state / use --load_4bit 1)", flush=True)
        raise
    (out / "qlora_config.json").write_text(json.dumps(
        {"load_4bit": bool(a.load_4bit), "quant": "nf4+double_quant+bf16_compute", "skip_quant": qlora.QUANT_SKIP,
         "init_from": a.init_from, "argv": sys.argv[1:]}, indent=1), encoding="utf-8")
    if a.dev and "model" in holder:
        dev_eval(holder["model"], kt, a, out)


def dev_eval(model, kt, a, out):
    """Teacher-forced dev scoring with the in-memory model: mean NLL and accuracy per question type."""
    import math
    import torch
    from kev.data import load_records, materialize
    from kev.model import load_tokenizer
    tok = load_tokenizer(a.base)
    recs = load_records(a.dev)
    if a.dev_limit:
        recs = recs[: a.dev_limit]
    model.eval()
    stats = {}
    t0 = time.time()
    with torch.no_grad():
        for r in recs:
            rec = materialize(r)
            if len(tok(rec["state"])["input_ids"]) > a.max_state:
                continue
            enc = model.encode(tok, rec, max_state=a.max_state, max_branch=1024, strict=False)
            for p, q in zip(model.probs(enc), rec["questions"]):
                s = stats.setdefault(q["qtype"], {"n": 0, "correct": 0, "nll": 0.0})
                s["n"] += 1; s["correct"] += int(int(p.argmax()) == q["label"]); s["nll"] += -math.log(max(float(p[q["label"]]), 1e-9))
    res = {t: {"n": s["n"], "acc": s["correct"] / s["n"], "nll": s["nll"] / s["n"]} for t, s in stats.items()}
    line = {"event": "dev", "seconds": round(time.time() - t0, 1), "by_type": res}
    print("dev=" + json.dumps(line), flush=True)
    with (out / "train_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
