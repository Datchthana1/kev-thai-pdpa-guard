"""Optuna hyperparameter search for the Kev-4B QLoRA fine-tune: find the settings with the LOWEST DEV LOSS.

Why dev loss and not train loss: the lowest train loss is reached by memorising the training set (overfitting). The
point we want at the bottom of the loss landscape is the one that also holds on records the model has never seen, so
every trial is scored by the mean negative log-likelihood (cross-entropy) on held-out dev records.

One full run takes ~3 h on an RTX 4060 8 GB, so trials are short (``--trial_steps`` optimizer steps on a shuffled
subset) and bad trials are pruned early: each trial streams its per-step training loss, the running mean is reported to
Optuna, and the MedianPruner stops trials that are clearly worse than earlier ones at the same step. Every trial runs in
its own process (scripts/train.py), so GPU memory is fully released between trials. The study is stored in SQLite and
can be stopped and resumed at any time.

    python scripts/tune_optuna.py --trials 12 --trial_steps 40            # search (~25 min per completed trial)
    python scripts/tune_optuna.py --report                                 # best params + history, no training
    bash scripts/run_pipeline.sh  # then retrain with the flags printed in results/optuna/best_params.json

Search space (Kev's own training knobs; LoRA rank is fixed at 16 because we warm-start from jaredpalmer/kev-4b):
    lr              log-uniform 5e-6 .. 1e-4
    accum           4, 8, 16           (effective batch)
    regulariser     none | label_smoothing 0.02..0.15 | brier_w 0.1..1.0   (Kev allows only one at a time)
    ord_w           0 .. 1             (ranked-probability term for the 0-4 risk score)
    p_none_pair     0 | 0.25           ("none of the above" minimal pairs)
    weight_decay    0 .. 0.1
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "results" / "optuna"
STEP_RE = re.compile(r"step=(\d+)/(\d+) loss=([0-9.]+)")


def subset(src: Path, dst: Path, n: int, seed: int) -> Path:
    """Write a fixed random subset of a JSONL file (same subset for every trial, so trials are comparable)."""
    lines = src.read_text(encoding="utf-8").splitlines()
    random.Random(seed).shuffle(lines)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines[:n]) + "\n", encoding="utf-8")
    return dst


def suggest(trial: optuna.Trial) -> dict:
    p = {
        "lr": trial.suggest_float("lr", 5e-6, 1e-4, log=True),
        "accum": trial.suggest_categorical("accum", [4, 8, 16]),
        "ord_w": trial.suggest_float("ord_w", 0.0, 1.0),
        "p_none_pair": trial.suggest_categorical("p_none_pair", [0.0, 0.25]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }
    reg = trial.suggest_categorical("regulariser", ["none", "label_smoothing", "brier_w"])
    if reg == "label_smoothing":
        p["label_smoothing"] = trial.suggest_float("label_smoothing", 0.02, 0.15)
    elif reg == "brier_w":
        p["brier_w"] = trial.suggest_float("brier_w", 0.1, 1.0)
    return p


def flags(p: dict) -> list[str]:
    """Hyperparameters -> scripts/train.py / kev.train flags."""
    out = ["--lr", f"{p['lr']:.3e}", "--accum", str(p["accum"]), "--ord_w", f"{p['ord_w']:.3f}",
           "--p_none_pair", str(p["p_none_pair"]), "--weight_decay", f"{p['weight_decay']:.4f}"]
    for k in ("label_smoothing", "brier_w"):
        if k in p:
            out += [f"--{k}", f"{p[k]:.4f}"]
    return out


def dev_nll(log_path: Path) -> float:
    """Question-weighted mean dev NLL from the {"event": "dev"} line that scripts/train.py appends."""
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        if '"event": "dev"' in line:
            by = json.loads(line)["by_type"]
            n = sum(v["n"] for v in by.values())
            return sum(v["nll"] * v["n"] for v in by.values()) / n
    raise RuntimeError(f"no dev line in {log_path}")


def make_objective(a, train_file: Path):
    def objective(trial: optuna.Trial) -> float:
        p = suggest(trial)
        run = OUT / "runs" / f"trial-{trial.number:03d}"
        cmd = [PY, str(ROOT / "scripts" / "train.py"), "--data", str(train_file), "--dev", a.dev, "--dev_limit", str(a.dev_limit),
               "--init_from", "jaredpalmer/kev-4b", "--load_4bit", "1", "--epochs", "1", "--max_steps", str(a.trial_steps),
               "--batch", "1", "--max_state", "1024", "--shared_prefix", "1", "--length_sort", "1", "--seed", "1",
               "--out", str(run)] + flags(p)
        print(f"\n[trial {trial.number}] {json.dumps(p)}", flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HOME=os.environ.get("HF_HOME", "D:/hf_cache"))
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        losses = []
        for line in proc.stdout:
            m = STEP_RE.search(line)
            if m:
                step, loss = int(m.group(1)), float(m.group(3))
                losses.append(loss)
                running = sum(losses[-10:]) / len(losses[-10:])
                trial.report(running, step)
                print(f"  step {step}/{a.trial_steps} loss {loss:.4f} running10 {running:.4f}", flush=True)
                if trial.should_prune():
                    proc.kill()
                    proc.wait()
                    print(f"  pruned at step {step} (running loss {running:.4f})", flush=True)
                    raise optuna.TrialPruned()
            elif "Traceback" in line or "OOM" in line or "Error" in line:
                print("  " + line.rstrip(), flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"trial {trial.number} failed (exit {proc.returncode})")
        value = dev_nll(run / "train_log.jsonl")
        trial.set_user_attr("minutes", round((time.time() - t0) / 60, 1))
        trial.set_user_attr("final_train_running10", sum(losses[-10:]) / max(len(losses[-10:]), 1))
        for f in run.glob("*.safetensors"):   # keep the log, drop the weights (disk)
            f.unlink()
        print(f"  dev NLL = {value:.4f}  ({trial.user_attrs['minutes']} min)", flush=True)
        return value
    return objective


def report(study: optuna.Study) -> None:
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        print("no completed trials yet")
        return
    best = study.best_trial
    best_flags = flags({**best.params, **{k: best.params[k] for k in ("label_smoothing", "brier_w") if k in best.params}})
    summary = {"best_trial": best.number, "best_dev_nll": best.value, "params": best.params, "train_flags": best_flags,
               "n_complete": len(done), "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "best_params.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    rows = ["| trial | state | dev NLL | lr | accum | regulariser | ord_w | minutes |", "|---|---|---|---|---|---|---|---|"]
    for t in study.trials:
        pr = t.params
        reg = pr.get("regulariser", "")
        if reg in pr:
            reg += f" {pr[reg]:.3f}"
        rows.append(f"| {t.number} | {t.state.name.lower()} | {t.value if t.value is None else round(t.value, 4)} | "
                    f"{pr.get('lr', 0):.2e} | {pr.get('accum')} | {reg} | {pr.get('ord_w', 0):.2f} | {t.user_attrs.get('minutes', '')} |")
    (OUT / "trials.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [t.number for t in done]
        ys = [t.value for t in done]
        best_so_far = [min(ys[: i + 1]) for i in range(len(ys))]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.scatter(xs, ys, s=18, label="trial dev NLL")
        ax.plot(xs, best_so_far, label="best so far")
        ax.set_xlabel("trial")
        ax.set_ylabel("dev NLL (lower is better)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "optuna_history.png", dpi=150)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(json.dumps(summary, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--trial_steps", type=int, default=40, help="optimizer steps per trial")
    ap.add_argument("--train_subset", type=int, default=1200, help="records sampled from train for the search")
    ap.add_argument("--train", default=str(ROOT / "data" / "processed" / "kev" / "train.jsonl"))
    ap.add_argument("--dev", default=str(ROOT / "data" / "processed" / "kev" / "dev.jsonl"))
    ap.add_argument("--dev_limit", type=int, default=150)
    ap.add_argument("--study", default="kev-pdpa-4b")
    ap.add_argument("--report", action="store_true", help="only print/save the current best, no training")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=a.study, storage=f"sqlite:///{(OUT / 'study.db').as_posix()}", load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=4),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=max(5, a.trial_steps // 3)))
    if not a.report:
        if not study.trials:   # first trial = the current recipe, so the search starts from a known baseline
            study.enqueue_trial({"lr": 2e-5, "accum": 8, "ord_w": 0.0, "p_none_pair": 0.0, "weight_decay": 0.01, "regulariser": "none"})
        train_file = subset(Path(a.train), OUT / "train_subset.jsonl", a.train_subset, seed=1)
        study.optimize(make_objective(a, train_file), n_trials=a.trials, catch=(RuntimeError,))
    report(study)
    return 0


if __name__ == "__main__":
    sys.exit(main())
