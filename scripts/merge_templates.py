"""Merge data/templates/parts/{A*,B*}_*.jsonl into train_A.jsonl / test_B.jsonl with T-A-### / T-B-### ids."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "data" / "templates"
for letter in ("A", "B"):
    rows = []
    for f in sorted((root / "parts").glob(f"{letter}*_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    for i, r in enumerate(rows, 1):
        r["template_id"] = f"T-{letter}-{i:03d}"
    name = "train_A.jsonl" if letter == "A" else "test_B.jsonl"
    with open(root / name, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))
