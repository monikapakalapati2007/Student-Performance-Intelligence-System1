"""Upload the trained model and its dataset to Hugging Face.

    python src/upload_huggingface.py --dry-run
    python src/upload_huggingface.py --repo-id <user>/air-quality-slm --dataset-repo-id <user>/air-quality-dataset

Never uploads anything without --repo-id, and never invents a token: it uses the
standard Hugging Face credentials (`huggingface-cli login`) or the HF_TOKEN
environment variable. Install the client first:  pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

MODEL_CARD = """---
library_name: pytorch
tags:
  - air-quality
  - small-language-model
  - causal-lm
  - gguf
  - ollama
language:
  - en
---

# Air Quality SLM — {n_params} parameter decoder-only model

A compact causal Transformer trained **from scratch** on air quality measurement
records. No pretrained weights were used and no public text corpus was trained on.

## Architecture

| property | value |
|---|---|
| type | decoder-only causal Transformer (`llama`-compatible) |
| layers | {n_layer} |
| heads | {n_head} |
| model width | {n_embd} |
| context length | {block_size} |
| vocabulary | {vocab_size} (byte-level BPE, learned from this corpus) |
| normalisation | RMSNorm (pre-norm) |
| positional encoding | Rotary Position Embeddings (RoPE) |
| feed-forward | SwiGLU |
| parameters | {n_params} |

## Files

- `config.json` — model hyperparameters
- `training_config.json` — optimiser, schedule and hardware used
- `tokenizer/` — `vocab.json`, `merges.json`, `tokenizer_config.json`
- `best_model.pt` — lowest validation loss checkpoint
- `final_model.pt` — last checkpoint of the run
- `air-quality-slm.f16.gguf` + `Modelfile` — for Ollama (if exported)

## Evaluation

Test split, untouched during training:

| metric | value |
|---|---|
| cross-entropy loss | {test_loss} |
| perplexity | {test_ppl} |
| token accuracy | {test_acc} |
| QA token F1 | {qa_f1} |
| QA numeric recall | {qa_num} |

## Loading

```python
import sys, torch
sys.path.append("src")              # from the project repository
from model import load_model_from_checkpoint
from tokenizer import ByteLevelBPETokenizer

model, cfg, payload = load_model_from_checkpoint("best_model.pt")
tok = ByteLevelBPETokenizer.load("tokenizer")
```

This is a **custom architecture**, not a `transformers` class, so
`AutoModel.from_pretrained()` will not load it — use the code above, or run the
GGUF build through Ollama.

## Limitations

Generated numbers can be wrong; numeric recall is reported above for that reason.
The model has no retrieval and cannot answer about records it has not seen in the
prompt. It is a demonstration-scale model, not a production air quality system.
"""

DATASET_CARD = """---
task_categories:
  - text-generation
tags:
  - air-quality
  - tabular-to-text
---

# Air Quality dataset (serialised for language modelling)

Source tables and their text serialisation used to train the Air Quality SLM.

## Splits

{split_table}

## Serialisation

Each measurement row is rendered as a record card plus deterministic
question-answer pairs whose answers quote that row's own values. Missing fields
are omitted rather than imputed; outliers are retained. See
`src/preprocess_air_quality.py` in the project repository.
"""


def existing_files(models_dir: str, roots: List[str]) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        bucket: List[str] = []
        for current, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "__"))]
            for name in sorted(filenames):
                if name.startswith(".") or name.endswith((".log", ".pyc")):
                    continue
                path = os.path.join(current, name)
                bucket.append(os.path.relpath(path, PROJECT_ROOT))
        if bucket:
            found[os.path.relpath(root, PROJECT_ROOT)] = bucket
    return found


def build_cards(project_root: str) -> Dict[str, str]:
    from utils import load_json

    models_dir = os.path.join(project_root, "models", "air_quality_slm")
    reports_dir = os.path.join(project_root, "reports")
    config = load_json(os.path.join(models_dir, "config.json"))
    training = load_json(os.path.join(models_dir, "training_config.json"))
    evaluation = None
    path = os.path.join(reports_dir, "evaluation_results.json")
    if os.path.exists(path):
        evaluation = load_json(path)
    metrics = (evaluation or {}).get("test_metrics", {})
    qa = (evaluation or {}).get("qa_generation", {})
    model_card = MODEL_CARD.format(
        n_params=f"{training.get('n_params', 0):,}",
        n_layer=config.get("n_layer"), n_head=config.get("n_head"),
        n_embd=config.get("n_embd"), block_size=config.get("block_size"),
        vocab_size=config.get("vocab_size"),
        test_loss=metrics.get("loss", "n/a"), test_ppl=metrics.get("perplexity", "n/a"),
        test_acc=metrics.get("token_accuracy", "n/a"), qa_f1=qa.get("mean_token_f1", "n/a"),
        qa_num=qa.get("mean_numeric_recall", "n/a"),
    )
    split_rows = []
    for name in ("train", "validation", "test"):
        path = os.path.join(project_root, "data", "processed", "aq_slm", f"{name}.jsonl")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                count = sum(1 for _ in fh)
            split_rows.append(f"| {name} | {count:,} documents |")
    split_table = "\n".join(["| split | size |", "|---|---|"] + split_rows) if split_rows else "| n/a | n/a |"
    return {"model": model_card, "dataset": DATASET_CARD.format(split_table=split_table)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Upload the SLM and its dataset to Hugging Face.")
    parser.add_argument("--repo-id", default=None, help="model repo, e.g. username/air-quality-slm")
    parser.add_argument("--dataset-repo-id", default=None, help="dataset repo, e.g. username/air-quality-dataset")
    parser.add_argument("--models-dir", default=os.path.join(PROJECT_ROOT, "models", "air_quality_slm"))
    parser.add_argument("--processed-dir", default=os.path.join(PROJECT_ROOT, "data", "processed", "aq_slm"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="list what would be uploaded")
    args = parser.parse_args(argv)

    if not os.path.exists(os.path.join(args.models_dir, "config.json")):
        print(f"ERROR: no trained model in {args.models_dir}. Run: python src/train_slm.py")
        return 2

    model_files = existing_files(args.models_dir, [args.models_dir,
                                                   os.path.join(PROJECT_ROOT, "models", "ollama")])
    data_files = existing_files(args.processed_dir, [args.processed_dir])
    print("Model artifacts to upload:")
    for root, files in model_files.items():
        print(f"  {root}/  ({len(files)} files)")
        for name in files[:12]:
            print(f"    - {name}")
    print("Dataset artifacts to upload:")
    for root, files in data_files.items():
        print(f"  {root}/  ({len(files)} files)")
        for name in files:
            print(f"    - {name}")

    if args.dry_run or not (args.repo_id or args.dataset_repo_id):
        print("\nDry run only. Re-run with --repo-id (and optionally --dataset-repo-id) to upload.")
        print("Authenticate first:  pip install huggingface_hub  &&  huggingface-cli login")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed.  pip install huggingface_hub")
        return 2

    api = HfApi()
    cards = build_cards(PROJECT_ROOT)
    if args.repo_id:
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)
        card_path = os.path.join(args.models_dir, "README.md")
        with open(card_path, "w", encoding="utf-8") as fh:
            fh.write(cards["model"])
        api.upload_folder(folder_path=args.models_dir, repo_id=args.repo_id, repo_type="model")
        ollama_dir = os.path.join(PROJECT_ROOT, "models", "ollama")
        if os.path.isdir(ollama_dir):
            api.upload_folder(folder_path=ollama_dir, repo_id=args.repo_id, repo_type="model",
                              path_in_repo="ollama")
        print(f"Model uploaded: https://huggingface.co/{args.repo_id}")
    if args.dataset_repo_id:
        api.create_repo(args.dataset_repo_id, repo_type="dataset", exist_ok=True, private=args.private)
        card_path = os.path.join(args.processed_dir, "README.md")
        with open(card_path, "w", encoding="utf-8") as fh:
            fh.write(cards["dataset"])
        api.upload_folder(folder_path=args.processed_dir, repo_id=args.dataset_repo_id,
                          repo_type="dataset")
        print(f"Dataset uploaded: https://huggingface.co/datasets/{args.dataset_repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
