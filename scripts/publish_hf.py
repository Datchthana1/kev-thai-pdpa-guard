"""Publish the fine-tuned adapter (model repo) and the synthetic dataset (dataset repo) to the Hugging Face Hub.

    python scripts/publish_hf.py --run runs/pdpa-4b [--model Datchthana/kev-thai-pdpa-guard]
                                 [--dataset Datchthana/thai-pdpa-pii-synthetic] [--revision-tag v1] [--dry-run]

Model repo gets: adapter_model.safetensors, adapter_config.json, head.pt (pointer head + calibration temperature),
tokenizer files, training/qlora configs and logs, MODEL_CARD.md as README.md, results/ and the architecture diagrams.
Dataset repo gets: data/processed/{train,dev,test}.jsonl (with meta), kev/*.jsonl (training format), QA report,
samples, FROZEN.sha256 and docs/DATASET_CARD.md as README.md. Uses the token from `hf auth login`.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
RUN_FILES = ["adapter_model.safetensors", "adapter_config.json", "head.pt", "tokenizer.json", "tokenizer_config.json",
             "chat_template.jinja", "training_config.json", "training_metrics.json", "qlora_config.json", "train_log.jsonl"]


def stage_model(run: Path, dst: Path) -> None:
    for name in RUN_FILES:
        if (run / name).exists():
            shutil.copy2(run / name, dst / name)
    shutil.copy2(ROOT / "MODEL_CARD.md", dst / "README.md")
    for sub, pattern in (("results", "*.md"), ("results", "*.png"), ("results", "eval_*.json"),
                         ("docs/architecture", "*.html"), ("results/optuna", "*")):
        src = ROOT / sub
        if src.exists():
            for f in src.glob(pattern):
                if f.is_file() and not f.name.endswith((".db", "train_subset.jsonl")):
                    (dst / sub).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst / sub / f.name)
    (dst / "src").mkdir()
    shutil.copytree(ROOT / "src" / "pdpa_guard", dst / "src" / "pdpa_guard", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "src" / "kev_qlora", dst / "src" / "kev_qlora", ignore=shutil.ignore_patterns("__pycache__"))


def stage_dataset(dst: Path) -> None:
    proc = ROOT / "data" / "processed"
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "QA_REPORT.md", "samples.md", "FROZEN.sha256"):
        shutil.copy2(proc / name, dst / name)
    (dst / "kev").mkdir()
    for f in (proc / "kev").glob("*.jsonl"):
        shutil.copy2(f, dst / "kev" / f.name)
    shutil.copy2(ROOT / "docs" / "DATASET_CARD.md", dst / "README.md")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/pdpa-4b")
    ap.add_argument("--model", default="Datchthana/kev-thai-pdpa-guard")
    ap.add_argument("--dataset", default="Datchthana/thai-pdpa-pii-synthetic")
    ap.add_argument("--revision-tag", default="", help="also tag the model repo, e.g. v1")
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="stage files and list them, upload nothing")
    a = ap.parse_args()
    api = HfApi()
    print("hf user:", api.whoami()["name"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model").mkdir()
        stage_model(ROOT / a.run, tmp / "model")
        if not a.skip_dataset:
            (tmp / "dataset").mkdir()
            stage_dataset(tmp / "dataset")
        for kind in ("model", "dataset"):
            if (tmp / kind).exists():
                files = sorted(p.relative_to(tmp / kind).as_posix() for p in (tmp / kind).rglob("*") if p.is_file())
                print(f"{kind}: {len(files)} files, {sum((tmp / kind / f).stat().st_size for f in files) / 2**20:.1f} MB")
        if a.dry_run:
            return 0
        api.create_repo(a.model, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(folder_path=tmp / "model", repo_id=a.model, repo_type="model",
                          commit_message=f"kev-thai-pdpa-guard {a.revision_tag or 'update'}: adapter, head, card, results")
        if a.revision_tag:
            api.create_tag(a.model, tag=a.revision_tag, repo_type="model", exist_ok=True)
        print(f"model -> https://huggingface.co/{a.model}")
        if not a.skip_dataset:
            api.create_repo(a.dataset, repo_type="dataset", exist_ok=True, private=False)
            api.upload_folder(folder_path=tmp / "dataset", repo_id=a.dataset, repo_type="dataset",
                              commit_message="thai-pdpa-pii-synthetic: frozen splits, QA report, card")
            print(f"dataset -> https://huggingface.co/datasets/{a.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
