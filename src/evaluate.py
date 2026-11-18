"""Test-set evaluation for the trained Air Quality SLM.

    python src/evaluate.py

Recomputes metrics from the saved artifacts only: the tokenizer and best
checkpoint in models/air_quality_slm/, the frozen test corpus in
data/processed/aq_slm/test.jsonl and the reference QA pairs in test_qa.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from model import load_model_for_inference  # noqa: E402
from tokenizer import create_blocks  # noqa: E402
from utils import ensure_dir, save_json, set_seed  # noqa: E402


def build_evaluation(
    test_metrics: Dict[str, float],
    samples: Sequence[Dict[str, Any]],
    config,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the evaluation_results.json payload from measured numbers only."""
    scored = [s for s in samples if s.get("has_reference")]
    free_form = [s for s in samples if not s.get("has_reference")]
    numeric = [s["numeric_recall"] for s in scored if s.get("numeric_recall") is not None]
    return {
        "test_metrics": {
            "loss": test_metrics.get("loss"),
            "perplexity": test_metrics.get("perplexity"),
            "token_accuracy": test_metrics.get("token_accuracy"),
            "scored_tokens": test_metrics.get("tokens"),
        },
        "qa_generation": {
            "held_out_questions": len(scored),
            "mean_token_f1": round(sum(s["token_f1"] for s in scored) / len(scored), 4) if scored else None,
            "exact_match_rate": round(sum(s["exact_match"] for s in scored) / len(scored), 4) if scored else None,
            "mean_numeric_recall": round(sum(numeric) / len(numeric), 4) if numeric else None,
            "numeric_questions": len(numeric),
            "free_form_questions": len(free_form),
            "generated_numerals_unsupported": sum(
                len(s.get("numerals_in_prediction_not_in_reference") or []) for s in scored
            ),
        },
        "model": {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
            "vocab_size": config.vocab_size,
            **(meta or {}),
        },
        "samples": list(samples),
    }


def evaluate_test_blocks(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss, tokens, correct = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += float(loss.item()) * y.numel()
            tokens += y.numel()
            correct += int((logits.argmax(-1) == y).sum().item())
    if tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "token_accuracy": float("nan"), "tokens": 0}
    mean = total_loss / tokens
    return {
        "loss": round(mean, 6),
        "perplexity": round(math.exp(min(mean, 20)), 4),
        "token_accuracy": round(correct / tokens, 6),
        "tokens": tokens,
    }


def load_texts(path: str) -> List[str]:
    texts = []
    if not os.path.exists(path):
        return texts
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def load_qa(path: str, limit: int) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained Air Quality SLM on the test split.")
    parser.add_argument("--models-dir", default=os.path.join(PROJECT_ROOT, "models", "air_quality_slm"))
    parser.add_argument("--reports-dir", default=os.path.join(PROJECT_ROOT, "reports"))
    parser.add_argument("--processed-dir", default=os.path.join(PROJECT_ROOT, "data", "processed", "aq_slm"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-questions", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1337)
    return _run(parser.parse_args(argv))


def _run(args) -> int:
    set_seed(args.seed)
    try:
        model, tokenizer, config, meta = load_model_for_inference(args.models_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2
    device = torch.device(meta["device"])
    model.to(device)
    print(f"Model: {meta['weights_path']} | device={device.type} | step={meta.get('step')}")

    test_texts = load_texts(os.path.join(args.processed_dir, "test.jsonl"))
    if not test_texts:
        print("ERROR: data/processed/aq_slm/test.jsonl missing. Run python src/train_slm.py first.")
        return 2

    from train_slm import BlockDataset

    token_ids = tokenizer.encode_documents(test_texts)
    blocks = create_blocks(token_ids, config.block_size)
    loader = DataLoader(BlockDataset(blocks), batch_size=args.batch_size, shuffle=False, drop_last=False)
    metrics = evaluate_test_blocks(model, loader, device)
    print(f"TEST | loss {metrics['loss']:.4f} | perplexity {metrics['perplexity']:.2f} | "
          f"token accuracy {metrics['token_accuracy']:.4f} | {metrics['tokens']:,} tokens")

    import inference as inf

    qa_records = load_qa(os.path.join(args.processed_dir, "test_qa.jsonl"), args.num_questions)
    samples = inf.run_qa_probes(model, tokenizer, device, qa_records,
                                extra_questions=inf.STANDARD_QUESTIONS)
    scored = [s for s in samples if s["has_reference"]]
    for s in samples[:6]:
        print(f"\nQ: {s['question']}")
        if s["has_reference"]:
            print(f"   reference: {s['reference']}")
        print(f"   generated: {s['prediction']}")
    if scored:
        print(f"\nMean token F1 over {len(scored)} held-out QA pairs: "
              f"{sum(s['token_f1'] for s in scored) / len(scored):.4f}")

    evaluation = build_evaluation(metrics, samples, config,
                                  {"weights": meta["weights_path"], "n_params": config.estimated_parameters()})
    ensure_dir(args.reports_dir)
    save_json(evaluation, os.path.join(args.reports_dir, "evaluation_results.json"))
    save_json(samples, os.path.join(args.reports_dir, "sample_predictions.json"))
    print(f"\nSaved {os.path.join(args.reports_dir, 'evaluation_results.json')}")
    print(f"Saved {os.path.join(args.reports_dir, 'sample_predictions.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
