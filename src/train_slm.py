"""End-to-end Air Quality SLM pipeline.

    python src/train_slm.py

Runs, in order: hardware detection -> dataset discovery & validation ->
preprocessing -> tokenizer training -> block creation -> model init ->
sanity checks -> full training -> validation-based model selection ->
test evaluation -> sample inference -> reports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dataset_loader as dl  # noqa: E402
import preprocess_air_quality as prep  # noqa: E402
from profiles import get_profile, write_profile_artifact  # noqa: E402
from model import (  # noqa: E402
    ModelConfig,
    PRESETS,
    build_config,
    build_model,
    load_checkpoint,
    save_checkpoint,
    save_config,
    select_preset,
)
from tokenizer import ByteLevelBPETokenizer, create_blocks  # noqa: E402
from utils import (  # noqa: E402
    Timer,
    count_parameters,
    describe_hardware,
    ensure_dir,
    format_seconds,
    get_hardware_info,
    save_json,
    set_seed,
    setup_logger,
)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class BlockDataset(Dataset):
    """Fixed-length causal LM blocks: input = tokens[:-1], target = tokens[1:]."""

    def __init__(self, blocks: List[List[int]]) -> None:
        self.blocks = [torch.tensor(b, dtype=torch.long) for b in blocks if len(b) > 1]

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        block = self.blocks[index]
        return block[:-1], block[1:]


def build_loader(blocks: List[List[int]], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        BlockDataset(blocks),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(blocks) > batch_size,
        num_workers=0,
    )


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """Mean cross-entropy, perplexity and teacher-forced token accuracy."""
    model.eval()
    total_loss, total_tokens, correct = 0.0, 0, 0
    for batch_index, (x, y) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        total_loss += float(loss.item()) * y.numel()
        total_tokens += y.numel()
        correct += int((logits.argmax(dim=-1) == y).sum().item())
    model.train()
    if total_tokens == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "token_accuracy": float("nan"), "tokens": 0}
    mean_loss = total_loss / total_tokens
    return {
        "loss": round(mean_loss, 6),
        "perplexity": round(math.exp(min(mean_loss, 20)), 4),
        "token_accuracy": round(correct / total_tokens, 6),
        "tokens": total_tokens,
    }


def lr_at_step(step: int, max_steps: int, base_lr: float, warmup: int, min_lr: float) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
def run_sanity_checks(
    model: torch.nn.Module,
    config: ModelConfig,
    tokenizer: ByteLevelBPETokenizer,
    train_blocks: List[List[int]],
    device: torch.device,
    logger,
    checkpoint_dir: str,
) -> Dict[str, Any]:
    """Mechanical checks that must pass before full training starts."""
    results: Dict[str, Any] = {"checks": {}, "passed": False}
    results["checks"]["dataset_non_empty"] = {"ok": len(train_blocks) > 0,
                                              "detail": f"{len(train_blocks)} training blocks"}
    text = "PM2.5 is 85 micrograms per cubic metre in Delhi."
    roundtrip = tokenizer.decode(tokenizer.encode(text))
    results["checks"]["tokenizer_roundtrip"] = {"ok": roundtrip == text, "detail": f"'{roundtrip[:60]}'"}

    dataset = BlockDataset(train_blocks)
    x, y = dataset[0]
    results["checks"]["tensor_shapes"] = {
        "ok": x.shape == y.shape and x.shape[0] == config.block_size - 1 and bool((y[:-1] == x[1:]).all()),
        "detail": f"x={tuple(x.shape)} y={tuple(y.shape)} shift_ok={bool((y[:-1] == x[1:]).all())}",
    }
    if not all(c["ok"] for c in results["checks"].values()):
        return results

    batch_x = x.unsqueeze(0).to(device)
    batch_y = y.unsqueeze(0).to(device)

    model.train()
    logits, loss = model(batch_x, batch_y)
    results["checks"]["forward_pass"] = {
        "ok": tuple(logits.shape) == (1, config.block_size - 1, config.vocab_size) and torch.isfinite(loss),
        "detail": f"logits={tuple(logits.shape)} loss={float(loss.detach()):.4f}",
    }

    # Causal masking: position t must not be influenced by tokens after t.
    # This is evaluated with dropout DISABLED. With dropout active the prefix and
    # full-sequence passes draw different masks, so the comparison would measure
    # dropout noise instead of the attention mask.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prefix = batch_x[:, :16]
            logits_prefix, _ = model(prefix)
            logits_full, _ = model(batch_x)
            gap = float((logits_prefix - logits_full[:, :16]).abs().max())
        causal_ok = gap <= 1e-3
    finally:
        model.train(was_training)
    results["checks"]["causal_mask"] = {
        "ok": bool(causal_ok),
        "detail": ("prefix logits unchanged when future tokens are appended "
                   f"(max abs difference {gap:.2e}, dropout disabled)"),
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.zero_grad(set_to_none=True)
    _, loss = model(batch_x, batch_y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_finite = all(torch.isfinite(g).all() for g in grads)
    results["checks"]["backpropagation"] = {
        "ok": bool(grads) and bool(grad_finite),
        "detail": f"{len(grads)} tensors with grads, all finite={grad_finite}",
    }

    before = [p.detach().clone() for p in model.parameters() if p.requires_grad][:3]
    optimizer.step()
    after = [p.detach().clone() for p in model.parameters() if p.requires_grad][:3]
    changed = all(not torch.equal(b, a) for b, a in zip(before, after))
    results["checks"]["parameters_update"] = {"ok": bool(changed), "detail": "weights changed after one AdamW step"}

    # short overfit run: loss must decrease on a fixed batch
    torch.manual_seed(0)
    over_opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first_loss = last_loss = None
    losses = []
    for step in range(25):
        over_opt.zero_grad(set_to_none=True)
        _, step_loss = model(batch_x, batch_y)
        step_loss.backward()
        over_opt.step()
        losses.append(round(float(step_loss.item()), 4))
        if step == 0:
            first_loss = float(step_loss.item())
        last_loss = float(step_loss.item())
    decreased = last_loss is not None and first_loss is not None and last_loss < first_loss
    results["checks"]["loss_decreases"] = {
        "ok": bool(decreased),
        "detail": f"25 overfit steps on one batch: {first_loss:.4f} -> {last_loss:.4f}",
        "curve": losses,
    }

    # checkpoint round trip
    ensure_dir(checkpoint_dir)
    tmp_path = os.path.join(checkpoint_dir, "_sanity_check.pt")
    save_checkpoint(tmp_path, model, optimizer, config, step=1, epoch=0, best_val=None)
    payload = load_checkpoint(tmp_path, map_location=str(device))
    fresh = build_model(config)
    fresh.load_state_dict(payload["model_state"])
    fresh.to(device).eval()
    with torch.no_grad():
        reload_logits, _ = fresh(batch_x)
        model.eval()
        original_logits, _ = model(batch_x)
    same = torch.allclose(reload_logits, original_logits, atol=1e-5)
    results["checks"]["checkpoint_roundtrip"] = {
        "ok": bool(same), "detail": "reloaded checkpoint reproduces logits exactly"
    }
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    results["passed"] = all(c["ok"] for c in results["checks"].values())
    results["loss_curve_25_steps"] = losses
    model.train()
    return results


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Air Quality SLM end to end.")
    parser.add_argument("--data-dir", default=None,
                        help="Folder holding the tables (default: project data/ dirs).")
    parser.add_argument("--profile", default="auto",
                        choices=["auto", "air_quality", "student_performance"],
                        help="Dataset domain. 'auto' detects it from the files found.")
    parser.add_argument("--max-records", type=int, default=20000,
                        help="Cap on training rows (seeded subsample); 0 = use all.")
    parser.add_argument("--max-eval-records", type=int, default=2000,
                        help="Cap on validation/test rows (seeded subsample); 0 = use all.")
    parser.add_argument("--workdir", default=PROJECT_ROOT,
                        help="Root for models/, checkpoints/, reports/, data/processed/.")
    parser.add_argument("--preset", default=None, choices=sorted(PRESETS) + [None])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/latest_checkpoint.pt")
    parser.add_argument("--patience", type=int, default=2, help="Early-stopping patience in epochs.")
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--reports-only", action="store_true",
                        help="Regenerate the markdown reports from reports/run_metadata.json.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    workdir = os.path.abspath(args.workdir)
    reports_dir = os.path.join(workdir, "reports")
    ensure_dir(reports_dir)
    # models/, checkpoints/ and data/processed/ are named after the resolved
    # dataset profile, so two domains never overwrite each other's artifacts.

    logger = setup_logger("air_quality_slm", os.path.join(reports_dir, "train.log"))
    timer = Timer()
    set_seed(args.seed)

    if args.reports_only:
        from reporting import write_dataset_analysis, write_training_summary

        meta_path = os.path.join(reports_dir, "run_metadata.json")
        if not os.path.exists(meta_path):
            logger.error("No %s to regenerate reports from.", meta_path)
            return 2
        with open(meta_path, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        write_training_summary(saved, os.path.join(reports_dir, "training_summary.md"))
        write_dataset_analysis(saved, os.path.join(reports_dir, "dataset_analysis.md"))
        logger.info("Regenerated training_summary.md and dataset_analysis.md from %s", meta_path)
        return 0

    logger.info("=" * 78)
    logger.info("SLM - automated training pipeline")
    logger.info("=" * 78)

    # ---------------- 1. hardware ----------------
    hardware = get_hardware_info()
    logger.info("HARDWARE\n%s", describe_hardware(hardware))
    device = torch.device(args.device or hardware["selected_device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    if device.type == "mps" and not getattr(torch.backends, "mps", None):
        device = torch.device("cpu")
    timer.mark("hardware")
    save_json(hardware, os.path.join(reports_dir, "hardware.json"))

    # ---------------- 2. dataset discovery + validation ----------------
    search_dirs = [os.path.abspath(args.data_dir)] if args.data_dir else dl.default_search_dirs(workdir)
    logger.info("Searching for data in: %s", search_dirs)
    try:
        splits, discovery = dl.load_splits(search_dirs, seed=args.seed, profile_key=args.profile)
    except dl.NoAirQualityDataError as exc:
        logger.error("%s", exc)
        sys.stdout.flush()
        return 2

    profile = get_profile(discovery.get("profile") or args.profile)
    models_dir = os.path.join(workdir, "models", profile.model_dirname)
    tokenizer_dir = os.path.join(models_dir, "tokenizer")
    checkpoints_dir = os.path.join(workdir, "checkpoints")
    processed_dir = os.path.join(workdir, "data", "processed", profile.key)
    for path in (models_dir, tokenizer_dir, checkpoints_dir, processed_dir):
        ensure_dir(path)
    logger.info("Dataset profile: %s (%s)", profile.key, profile.description)
    if profile.key == "student_performance":
        logger.warning("Training on STUDENT PERFORMANCE data. The original brief specified "
                       "an air quality model; reports will name this dataset explicitly.")

    # A CPU budget cannot consume millions of rows, so cap the rows used per split
    # with a seeded, documented subsample (never re-split, never shuffled across
    # splits). Original files are untouched.
    original_rows = {k: int(len(v)) for k, v in splits.items()}
    rng = np.random.RandomState(args.seed)
    caps = {"train": args.max_records, "validation": args.max_eval_records,
            "test": args.max_eval_records}
    sampled: Dict[str, pd.DataFrame] = {}
    for name, df in splits.items():
        cap = caps.get(name)
        if cap and len(df) > cap:
            keep = sorted(rng.choice(len(df), cap, replace=False))
            sampled[name] = df.iloc[keep].reset_index(drop=True)
        else:
            sampled[name] = df.reset_index(drop=True)
    discovery["subsampling"] = {
        "seed": args.seed,
        "original_rows": original_rows,
        "rows_used": {k: int(len(v)) for k, v in sampled.items()},
        "note": ("Rows were subsampled with a fixed seed for CPU training. Splits were not "
                 "re-partitioned, so validation and test remain disjoint from each other.")
        if any(original_rows[k] > len(sampled[k]) for k in sampled) else None,
    }
    split_frames = {k: v for k, v in sampled.items()}
    for required in ("train", "validation", "test"):
        if required not in split_frames:
            logger.error("Split '%s' is missing. Found: %s", required, list(split_frames))
            return 2

    logger.info("Splits: %s  (source: %s)",
                {k: len(v) for k, v in split_frames.items()}, discovery.get("source"))

    analyses = {name: dl.analyze_frame(df, name) for name, df in split_frames.items()}
    overlap = dl.check_split_overlap(split_frames)
    spec = dl.classify_fields(split_frames["train"])
    for name, df in split_frames.items():
        extra = dl.classify_fields(df)
        for attr in ("pollutant", "weather"):
            for key, col in getattr(extra, attr).items():
                getattr(spec, attr).setdefault(key, col)
        for attr in ("category", "location", "numeric", "categorical"):
            current = getattr(spec, attr)
            for col in getattr(extra, attr):
                if col not in current:
                    current.append(col)

    if overlap["leakage_detected"]:
        logger.warning("Duplicate records across splits detected: %s", overlap["pairwise"])
    else:
        logger.info("No overlapping records between splits.")

    # ---------------- 3. preprocessing ----------------
    logger.info("Preparing %s text ...", profile.subject)
    prep_report, cleaned, spec = prep.prepare(split_frames, processed_dir, spec=spec, profile=profile)
    qa_sets = write_qa_sets(cleaned, spec, processed_dir)
    for name, info in prep_report["splits"].items():
        logger.info("  %-10s rows=%d docs=%d chars=%d",
                    name, info["rows_after_cleaning"], info["documents"], info["characters"])
    timer.mark("preprocess")

    # ---------------- 4. tokenizer ----------------
    train_docs = read_jsonl_texts(os.path.join(processed_dir, "train.jsonl"))
    validation_docs = read_jsonl_texts(os.path.join(processed_dir, "validation.jsonl"))
    test_docs = read_jsonl_texts(os.path.join(processed_dir, "test.jsonl"))
    if not train_docs:
        logger.error("No training documents were produced; aborting.")
        return 2

    # On resume the vocabulary must match the checkpoint exactly, so reuse the
    # saved tokenizer instead of retraining it on whatever data is present now.
    saved_tokenizer = os.path.join(tokenizer_dir, "merges.json")
    if args.resume and os.path.exists(saved_tokenizer):
        tokenizer = ByteLevelBPETokenizer.load(tokenizer_dir)
        logger.info("Resuming: loaded saved tokenizer (vocab %d)", tokenizer.vocab_size)
    else:
        logger.info("Training byte-level BPE tokenizer (target vocab %d) ...", args.vocab_size)
        tokenizer = ByteLevelBPETokenizer.train(train_docs, vocab_size=args.vocab_size, progress=not args.quiet)
        tokenizer.save(tokenizer_dir)
    train_tokens = tokenizer.encode_documents(train_docs)
    val_tokens = tokenizer.encode_documents(validation_docs)
    test_tokens = tokenizer.encode_documents(test_docs)
    logger.info("Vocabulary: %d | train tokens: %s | val tokens: %s | test tokens: %s",
                tokenizer.vocab_size, f"{len(train_tokens):,}", f"{len(val_tokens):,}", f"{len(test_tokens):,}")

    # ---------------- 5. model ----------------
    preset = select_preset(hardware, args.preset)
    preset_cfg = dict(PRESETS[preset])
    config = build_config(tokenizer.vocab_size, preset, overrides={
        "block_size": args.block_size, "n_layer": None, "n_head": None, "n_embd": None,
    })
    batch_size = args.batch_size or preset_cfg["batch_size"]
    grad_accum = args.grad_accum or preset_cfg["grad_accum"]
    epochs = args.epochs or preset_cfg["epochs"]
    max_steps = args.max_steps if args.max_steps is not None else preset_cfg["max_steps"]
    base_lr = args.lr or preset_cfg["lr"]
    precision = "float16" if device.type == "cuda" else "float32"

    train_blocks = create_blocks(train_tokens, config.block_size)
    val_blocks = create_blocks(val_tokens, config.block_size)
    test_blocks = create_blocks(test_tokens, config.block_size)
    if not train_blocks:
        logger.error("Corpus is shorter than one block (%d tokens).", config.block_size)
        return 2

    model = build_model(config).to(device)
    mem = config.estimate_memory_mb(batch_size)
    n_params = count_parameters(model)
    logger.info("Preset '%s' | device=%s | precision=%s", preset, device.type, precision)
    logger.info("Model: %d layers, %d heads, n_embd=%d, block=%d, dropout=%.2f",
                config.n_layer, config.n_head, config.n_embd, config.block_size, config.dropout)
    logger.info("Parameters: %s (%.2f MB fp32) | estimated training memory: %.1f MB",
                f"{n_params:,}", n_params * 4 / 1024 ** 2, mem["total_train_mb"])
    logger.info("Blocks: train=%d val=%d test=%d | batch=%d x accum=%d",
                len(train_blocks), len(val_blocks), len(test_blocks), batch_size, grad_accum)
    if n_params > 60_000_000 and device.type == "cpu":
        logger.warning("Large model on CPU - consider --preset cpu_tiny.")

    save_config(config, os.path.join(models_dir, "config.json"))
    write_profile_artifact(models_dir, profile)

    # ---------------- 6. sanity checks ----------------
    logger.info("Running sanity checks ...")
    sanity = run_sanity_checks(model, config, tokenizer, train_blocks, device, logger, checkpoints_dir)
    for name, check in sanity["checks"].items():
        logger.info("  [%s] %-22s %s", "PASS" if check["ok"] else "FAIL", name, check["detail"])
    save_json(sanity, os.path.join(reports_dir, "sanity_checks.json"))
    if not sanity["passed"]:
        logger.error("Sanity checks failed - full training aborted.")
        return 3
    logger.info("All sanity checks passed.")
    timer.mark("sanity")

    # ---------------- 7. training ----------------
    train_config = {
        "preset": preset,
        "device": device.type,
        "precision": precision,
        "seed": args.seed,
        "batch_size": batch_size,
        "grad_accum": max(1, grad_accum),
        "block_size": config.block_size,
        "max_steps": max_steps,
        "max_epochs": epochs,
        "learning_rate": base_lr,
        "min_lr": base_lr * 0.1,
        "warmup_steps": max(10, int(0.05 * max_steps)),
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "optimizer": "AdamW",
        "schedule": "linear warmup + cosine decay",
        "eval_interval": 50,
        "patience": args.patience,
        "tie_weights": config.tie_weights,
        "n_params": n_params,
        "estimated_train_memory_mb": mem["total_train_mb"],
        "vocab_size": tokenizer.vocab_size,
        "num_train_blocks": len(train_blocks),
        "num_val_blocks": len(val_blocks),
        "num_test_blocks": len(test_blocks),
    }
    save_json(train_config, os.path.join(models_dir, "training_config.json"))

    train_loader = build_loader(train_blocks, batch_size, shuffle=True)
    val_loader = build_loader(val_blocks, batch_size, shuffle=False)
    test_loader = build_loader(test_blocks, batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    autocast_dtype = torch.float16 if use_amp else None

    start_step, start_epoch, best_val = 0, 0, float("inf")
    latest_path = os.path.join(checkpoints_dir, "latest_checkpoint.pt")
    if args.resume and os.path.exists(latest_path):
        payload = load_checkpoint(latest_path, map_location=str(device))
        model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        start_step = int(payload.get("step", 0))
        start_epoch = int(payload.get("epoch", 0))
        best_val = float(payload.get("best_val") or float("inf"))
        logger.info("Resumed from %s at step %d (best val %.4f)", latest_path, start_step,
                    best_val if math.isfinite(best_val) else float("nan"))

    history: List[Dict[str, Any]] = []
    tail_losses: List[float] = []  # trailing window: epoch-end training loss
    step = start_step
    best_path = os.path.join(models_dir, "best_model.pt")
    epochs_without_improvement = 0
    stop_training = False
    initial_val = evaluate_loss(model, val_loader, device, args.eval_max_batches, autocast_dtype)
    logger.info("Initial validation: loss %.4f ppl %.2f", initial_val["loss"], initial_val["perplexity"])
    best_val = min(best_val, initial_val["loss"]) if history else initial_val["loss"]

    logger.info("Starting full training ...")
    train_start = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        if stop_training:
            break
        epoch_loss, epoch_steps, epoch_correct, epoch_tokens = 0.0, 0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for micro_step, (x, y) in enumerate(train_loader):
            if step >= max_steps:
                stop_training = True
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            lr = lr_at_step(step, max_steps, base_lr, train_config["warmup_steps"], train_config["min_lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr

            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
            scaled = loss / max(1, grad_accum)
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            tail_losses.append(float(loss.item()))
            tail_losses[:] = tail_losses[-50:]
            epoch_loss += float(loss.item()) * y.numel()
            epoch_tokens += y.numel()
            epoch_correct += int((logits.detach().argmax(dim=-1) == y).sum().item())
            epoch_steps += 1

            if (micro_step + 1) % max(1, grad_accum) == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % 10 == 0:
                    elapsed = time.perf_counter() - train_start
                    tok_per_sec = (step - start_step) * batch_size * grad_accum * config.block_size / max(elapsed, 1e-6)
                    logger.info("epoch %d | step %d/%d | loss %.4f | lr %.2e | %.0f tok/s | %s",
                                epoch + 1, step, max_steps, float(loss.item()), lr, tok_per_sec,
                                format_seconds(elapsed))

                if step % train_config["eval_interval"] == 0:
                    val = evaluate_loss(model, val_loader, device, args.eval_max_batches, autocast_dtype)
                    improved = val["loss"] < best_val
                    logger.info("  [eval] step %d | val loss %.4f | val ppl %.2f | val acc %.4f%s",
                                step, val["loss"], val["perplexity"], val["token_accuracy"],
                                "  <- best" if improved else "")
                    history.append({"kind": "eval", "step": step, "epoch": epoch + 1, "val": val})
                    if improved:
                        best_val = val["loss"]
                        save_checkpoint(best_path, model, optimizer, config, step, epoch + 1, best_val,
                                        {"val": val, "train_config": train_config})
                    save_checkpoint(latest_path, model, optimizer, config, step, epoch + 1, best_val,
                                    {"val": val})

        train_epoch_loss = epoch_loss / max(1, epoch_tokens)
        train_epoch_acc = epoch_correct / max(1, epoch_tokens)
        val = evaluate_loss(model, val_loader, device, args.eval_max_batches, autocast_dtype)
        improved = val["loss"] < best_val
        logger.info("Epoch %d/%d done | train loss %.4f acc %.4f | val loss %.4f ppl %.2f acc %.4f%s",
                    epoch + 1, epochs, train_epoch_loss, train_epoch_acc, val["loss"],
                    val["perplexity"], val["token_accuracy"], "  <- best" if improved else "")
        history.append({
            "kind": "epoch",
            "epoch": epoch + 1, "step": step,
            "train_loss": round(train_epoch_loss, 6),
            "train_loss_epoch_end": round(sum(tail_losses) / max(1, len(tail_losses)), 6),
            "train_token_accuracy": round(train_epoch_acc, 6),
            "val": val,
        })
        if improved:
            best_val = val["loss"]
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, optimizer, config, step, epoch + 1, best_val,
                            {"val": val, "train_config": train_config})
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping: validation loss has not improved for %d epochs.",
                            epochs_without_improvement)
                stop_training = True
        save_checkpoint(latest_path, model, optimizer, config, step, epoch + 1, best_val, {"val": val})

    training_seconds = time.perf_counter() - train_start
    timer.mark("train")
    logger.info("Training finished: %s, %d steps, best val loss %.4f",
                format_seconds(training_seconds), step - start_step, best_val)

    # ---------------- 8. final model ----------------
    final_path = os.path.join(models_dir, "final_model.pt")
    save_checkpoint(final_path, model, optimizer, config, step, epochs, best_val, {"train_config": train_config})
    logger.info("Saved final model -> %s", final_path)
    if not os.path.exists(best_path):
        # no validation improvement was ever recorded (e.g. very short run)
        save_checkpoint(best_path, model, optimizer, config, step, epochs, best_val,
                        {"val": {"loss": best_val}, "train_config": train_config})
        logger.info("No improved checkpoint existed; best_model.pt mirrors the final model.")
    logger.info("Saved best model  -> %s", best_path)

    # ---------------- 9. test evaluation ----------------
    logger.info("Evaluating on the untouched test split ...")
    test_model, test_config, _ = __import__("model").load_model_from_checkpoint(best_path, device=str(device))
    test_model.to(device)
    test_metrics = evaluate_loss(test_model, test_loader, device, args.eval_max_batches, autocast_dtype)
    logger.info("TEST | loss %.4f | perplexity %.2f | token accuracy %.4f",
                test_metrics["loss"], test_metrics["perplexity"], test_metrics["token_accuracy"])

    samples: List[Dict[str, Any]] = []
    if not args.skip_inference:
        logger.info("Running sample %s inference ...", profile.subject)
        import inference as inf

        samples = inf.run_qa_probes(
            test_model, tokenizer, device,
            qa_sets.get("test", [])[:12],
            extra_questions=profile.questions,
        )
        for item in samples:
            logger.info("  Q: %s\n     A: %s", item["question"], item["prediction"][:180])

    import evaluate as ev
    evaluation = ev.build_evaluation(
        test_metrics=test_metrics,
        samples=samples,
        config=config,
        meta={"weights": best_path, "n_params": n_params},
    )
    save_json(evaluation, os.path.join(reports_dir, "evaluation_results.json"))
    save_json(samples, os.path.join(reports_dir, "sample_predictions.json"))

    # ---------------- 10. reports ----------------
    timer.mark("total")
    ff_hidden = int(model.blocks[0].mlp.w1.weight.shape[0]) if config.n_layer else 0
    statistics = {
        "ff_hidden": ff_hidden,
        "correlation": dl.numeric_correlation(cleaned["train"], spec.numeric),
        "parameters": n_params,
        "weight_mb_fp32": round(n_params * 4 / 1024 ** 2, 2),
    }

    report = {
        "profile": {"key": profile.key, "display_name": profile.display_name,
                    "subject": profile.subject, "paper_title": profile.paper_title,
                    "description": profile.description,
                    "model_dirname": profile.model_dirname},
        "hardware": hardware,
        "statistics": statistics,
        "discovery": discovery,
        "analyses": analyses,
        "overlap": overlap,
        "preprocessing": prep_report,
        "field_spec": spec.to_dict(),
        "train_config": train_config,
        "config": config.to_dict(),
        "n_params": n_params,
        "history": history,
        "test_metrics": test_metrics,
        "training_seconds": round(training_seconds, 2),
        "timings": {k: round(v, 2) for k, v in timer.marks.items()},
        "samples": samples,
        "evaluation": evaluation,
        "paths": {
            "models_dir": models_dir,
            "best_model": best_path,
            "final_model": final_path,
            "tokenizer": tokenizer_dir,
            "checkpoints": checkpoints_dir,
        },
    }
    from reporting import write_dataset_analysis, write_training_summary

    write_training_summary(report, os.path.join(reports_dir, "training_summary.md"))
    write_dataset_analysis(report, os.path.join(reports_dir, "dataset_analysis.md"))
    save_json(report, os.path.join(reports_dir, "run_metadata.json"))

    logger.info("=" * 78)
    logger.info("DONE | test loss %.4f | test ppl %.2f | best val %.4f", test_metrics["loss"],
                test_metrics["perplexity"], best_val)
    logger.info("Artifacts: %s", models_dir)
    logger.info("Reports  : %s", reports_dir)
    logger.info("=" * 78)
    return 0


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def read_jsonl_texts(path: str) -> List[str]:
    texts: List[str] = []
    if not os.path.exists(path):
        return texts
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
    return texts


def write_qa_sets(cleaned: Dict[str, pd.DataFrame], spec, output_dir: str) -> Dict[str, List[Dict[str, str]]]:
    """Persist reference question/answer pairs per split for evaluation."""
    out: Dict[str, List[Dict[str, str]]] = {}
    for name, df in cleaned.items():
        records: List[Dict[str, str]] = []
        # QA probes are for evaluation/inference only, so a bounded sample is
        # enough and keeps the artifact small on a million-row dataset.
        key_column = spec.identifier[0] if spec.identifier else ""
        for _, row in df.head(2000).iterrows():
            context = prep.record_context(row, spec)
            key = str(row[key_column]).strip() if key_column else ""
            for question, answer in prep.qa_pairs_for_row(row, spec, max_pairs=2):
                records.append({"question": question, "answer": answer,
                                "context": context, "key": key})
        out[name] = records
        prep.write_jsonl(records, os.path.join(output_dir, f"{name}_qa.jsonl"))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
