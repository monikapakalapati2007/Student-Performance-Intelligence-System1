"""Air Quality SLM inference: prompted question answering and scoring.

    python src/inference.py                        # standard question set
    python src/inference.py --question "What is the PM2.5 level in Delhi?"
    python src/inference.py --interactive           # REPL
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch  # noqa: E402

from model import load_model_for_inference  # noqa: E402
from profiles import default_models_dir, read_model_profile  # noqa: E402

QA_HEADER = "### Question and answer"
Q_PREFIX = "Question:"
A_PREFIX = "Answer:"
DOC_END = "### End"
# Only markers that end a document. "\nQuestion:" is part of the prompt itself
# and must never be used, or generation stops before it produces anything.
STOP_STRINGS = (DOC_END, "\n###")

STANDARD_QUESTIONS: List[str] = [
    "What does PM2.5 represent?",
    "Which pollutants are present in the dataset?",
    "Which meteorological variables are recorded?",
    "Explain the relationship between particulate matter and weather conditions.",
    "What does the Air Quality Index summarise?",
]


def load_questions(models_dir: str) -> List[str]:
    """Profile-appropriate demo questions, saved beside the trained weights."""
    path = os.path.join(models_dir, "profile.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                questions = json.load(fh).get("questions") or []
            if questions:
                return list(questions)
        except Exception:
            pass
    return list(STANDARD_QUESTIONS)


def build_prompt(question: str, context: Optional[str] = None) -> str:
    """Prompt template that exactly mirrors the training document format."""
    text = f"{QA_HEADER}\n"
    if context:
        text += f"{context}\n"
    text += f"{Q_PREFIX} {question}\n{A_PREFIX}"
    return text


def _truncate_prompt(ids: List[int], block_size: int) -> List[int]:
    if len(ids) <= block_size - 8:
        return ids
    return ids[-(block_size - 8):]


@torch.no_grad()
def generate_answer(
    model: torch.nn.Module,
    tokenizer,
    question: str,
    device: torch.device,
    context: Optional[str] = None,
    max_new_tokens: int = 72,
    temperature: float = 0.2,
    top_k: int = 40,
) -> str:
    prompt_ids = tokenizer.encode(build_prompt(question, context), add_bos=True)
    prompt_ids = _truncate_prompt(prompt_ids, model.block_size)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
        stop_strings=STOP_STRINGS,
        decode_fn=tokenizer.decode,
        prompt_length=len(prompt_ids),
    )
    generated = tokenizer.decode(out[0, len(prompt_ids):].tolist())
    for stop in STOP_STRINGS:
        if stop in generated:
            generated = generated.split(stop)[0]
    return generated.strip()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
_NUM_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def score_answer(prediction: str, reference: str) -> Dict[str, Any]:
    """Token-F1, exact match and numeric support for a generated answer."""
    pred_tokens = _normalise(prediction).split()
    ref_tokens = _normalise(reference).split()
    if not pred_tokens or not ref_tokens:
        overlap = 0
    else:
        from collections import Counter

        common = Counter(pred_tokens) & Counter(ref_tokens)
        overlap = sum(common.values())
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ref_numbers = set(_NUM_PATTERN.findall(reference))
    pred_numbers = set(_NUM_PATTERN.findall(prediction))
    numeric_support = (len(ref_numbers & pred_numbers) / len(ref_numbers)) if ref_numbers else None
    if pred_numbers and not ref_numbers:
        numeric_support = None
    return {
        "token_f1": round(f1, 4),
        "exact_match": int(_normalise(prediction) == _normalise(reference)),
        "numeric_recall": None if numeric_support is None else round(numeric_support, 4),
        "numerals_in_prediction_not_in_reference": sorted(pred_numbers - ref_numbers),
    }


def run_qa_probes(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    qa_records: Sequence[Dict[str, str]],
    extra_questions: Optional[Sequence[str]] = None,
    temperature: float = 0.2,
    max_new_tokens: int = 72,
) -> List[Dict[str, Any]]:
    """Generate answers for held-out QA pairs plus free-form domain questions."""
    model.eval()
    results: List[Dict[str, Any]] = []
    for record in qa_records:
        question, reference = record["question"], record.get("answer", "")
        prediction = generate_answer(model, tokenizer, question, device,
                                     context=record.get("context") or None,
                                     temperature=temperature, max_new_tokens=max_new_tokens)
        item = {"question": question, "reference": reference, "prediction": prediction,
                "has_reference": True}
        item.update(score_answer(prediction, reference))
        results.append(item)
    for question in (extra_questions or []):
        prediction = generate_answer(model, tokenizer, question, device,
                                     temperature=temperature, max_new_tokens=max_new_tokens)
        results.append({
            "question": question, "reference": "", "prediction": prediction,
            "has_reference": False, "token_f1": None, "exact_match": None, "numeric_recall": None,
            "numerals_in_prediction_not_in_reference": [],
        })
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained Air Quality SLM.")
    parser.add_argument("--models-dir", default=default_models_dir(PROJECT_ROOT))
    parser.add_argument("--reports-dir", default=os.path.join(PROJECT_ROOT, "reports"))
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--question", default=None, help="Answer a single question and exit.")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--num-samples", type=int, default=8, help="Held-out QA pairs to probe.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--save", action="store_true", help="Write reports/sample_predictions.json")
    return parser.parse_args(argv)


def _load_test_qa(path: str, limit: int) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        model, tokenizer, config, meta = load_model_for_inference(args.models_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2
    device = torch.device(meta["device"])
    model.to(device)
    if args.processed_dir is None:
        profile_meta = read_model_profile(args.models_dir)
        args.processed_dir = os.path.join(PROJECT_ROOT, "data", "processed",
                                          profile_meta.get("key", "air_quality"))
    print(f"Loaded {meta['weights_path']} ({config.n_layer} layers, {config.n_embd} d_model, "
          f"vocab {config.vocab_size}) on {device.type}")
    print(f"Trained for {meta.get('step')} steps | best val loss {meta.get('best_val')}\n")

    if args.question:
        answer = generate_answer(model, tokenizer, args.question, device,
                                 temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        print(f"Q: {args.question}\nA: {answer}")
        return 0

    if args.interactive:
        print("Interactive mode. Empty line or Ctrl-C to quit.")
        while True:
            try:
                question = input("\nQ: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                break
            answer = generate_answer(model, tokenizer, question, device,
                                     temperature=args.temperature, max_new_tokens=args.max_new_tokens)
            print(f"A: {answer}")
        return 0

    qa_records = _load_test_qa(os.path.join(args.processed_dir, "test_qa.jsonl"), args.num_samples)
    results = run_qa_probes(model, tokenizer, device, qa_records,
                            extra_questions=load_questions(args.models_dir),
                            temperature=args.temperature, max_new_tokens=args.max_new_tokens)
    for item in results:
        print(f"Q: {item['question']}")
        if item["has_reference"]:
            print(f"   reference: {item['reference']}")
            print(f"   generated: {item['prediction']}")
            print(f"   token_f1={item['token_f1']} exact={item['exact_match']} "
                  f"numeric_recall={item['numeric_recall']}")
        else:
            print(f"   generated: {item['prediction']}")
        print()

    scored = [r for r in results if r["has_reference"]]
    if scored:
        mean_f1 = sum(r["token_f1"] for r in scored) / len(scored)
        exact = sum(r["exact_match"] for r in scored) / len(scored)
        print(f"Mean token F1 on {len(scored)} held-out questions: {mean_f1:.4f} | exact match: {exact:.4f}")

    if args.save:
        os.makedirs(args.reports_dir, exist_ok=True)
        path = os.path.join(args.reports_dir, "sample_predictions.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"Saved predictions -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
