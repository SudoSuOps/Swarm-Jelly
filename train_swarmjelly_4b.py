#!/usr/bin/env python3
"""SwarmJelly-4B — Self-Healing Intelligence Model
=====================================================

The Royal Jelly engine. Learns from 100K+ failure patterns to:
  1. Diagnose agent failures
  2. Generate repair strategies
  3. Classify failure modes (7 failure taxonomy)
  4. Multiply failures into 5x Royal Jelly training pairs
  5. Innovate new pair types for underrepresented domains

Failure Taxonomy (7 dominant LLM failure modes):
  reasoning/missing_step, reasoning/false_assumption
  knowledge/hallucination, knowledge/overgeneralization
  instruction/drift, instruction/schema_break
  agent/tool_misuse

Royal Jelly tasks per failure: DIAGNOSE, REPAIR, PREVENT, DETECT, COMPARE

Gold Standard config (4B tier): LR 2e-5, r=32, epoch 0.8
Hardware: RTX PRO 6000 Blackwell (96GB) — GPU 1 on swarmrails

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 train_swarmjelly_4b.py --smoke-test
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 train_swarmjelly_4b.py --pilot
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 train_swarmjelly_4b.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

MODEL_NAME = "Qwen/Qwen3.5-4B"
TRAIN_FILE = "/data2/swarmjelly-4b/swarmjelly_train.jsonl"
EVAL_FILE = "/data2/swarmjelly-4b/swarmjelly_eval.jsonl"
OUTPUT_DIR = Path("/data2/swarmjelly-4b/lora")
MERGED_DIR = Path("/data2/swarmjelly-4b/merged")
LOG_DIR = Path("/data2/swarmjelly-4b/logs")
BUILD_NAME = "SwarmJelly-4B"

# ═══════════════════════════════════════════════════════════════════════
# GOLD STANDARD HYPERPARAMETERS — 4B TIER
# ═══════════════════════════════════════════════════════════════════════

LORA_R = 32
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 2e-5
MAX_EPOCH_FRACTION = 0.8
BATCH_SIZE = 4
GRAD_ACCUM = 8                 # Effective batch = 32
MAX_SEQ_LEN = 4096
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"

EVAL_STEPS = 50
SAVE_STEPS = 100
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001
MAX_EVAL_SAMPLES = 2000
SAVE_TOTAL_LIMIT = 5

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def validate_data(train_path: str, eval_path: str):
    """Pre-flight data validation."""
    for label, path in [("Train", train_path), ("Eval", eval_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    sys_prompts = set()
    train_count = 0
    with open(train_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            if len(msgs) < 2:
                raise ValueError(f"Record at {train_path}:{i} has < 2 messages")
            if msgs[0].get("role") == "system":
                sys_prompts.add(msgs[0]["content"][:100])
            train_count += 1

    eval_count = sum(1 for line in open(eval_path) if line.strip())
    train_sha = hashlib.sha256(open(train_path, "rb").read()).hexdigest()
    eval_sha = hashlib.sha256(open(eval_path, "rb").read()).hexdigest()

    print(f"  Train: {train_count:,} records (SHA256: {train_sha[:16]}...)")
    print(f"  Eval:  {eval_count:,} records (SHA256: {eval_sha[:16]}...)")
    print(f"  System prompt diversity: {len(sys_prompts)} unique")

    if train_count < 1000:
        raise ValueError(f"Dataset too small: {train_count}")

    return {
        "train_count": train_count,
        "eval_count": eval_count,
        "train_sha256": train_sha,
        "eval_sha256": eval_sha,
        "system_prompt_diversity": len(sys_prompts),
    }


def push_model_lineage(manifest: dict, data_info: dict) -> dict | None:
    """Push model training lineage to hive-ledger.

    Uses a deterministic batch_id derived from the training data SHA-256
    (already in data_info). This can be reconciled with actual batch IDs later.

    Returns the API response dict, or None if push is skipped/fails.
    """
    import urllib.error
    import urllib.request

    ledger_url = os.environ.get("HIVE_LEDGER_URL", "https://ledger.swarmandbee.ai")
    admin_key = os.environ.get("HIVE_ADMIN_KEY", "")

    if not admin_key:
        print("  [ledger] HIVE_ADMIN_KEY not set — skipping model registration")
        return None

    # Derive batch_id from training data hash
    train_hash = data_info.get("train_sha256", "")
    if not train_hash:
        train_hash = hashlib.sha256(TRAIN_FILE.encode()).hexdigest()
    batch_id = f"BATCH-SJ4-{train_hash[:12]}"

    body = {
        "model_id": manifest["model"],
        "batch_id": batch_id,
        "training_run_id": f"run-{manifest.get('completed_at', '')[:10]}",
        "pairs_used": data_info.get("train_count", 0),
        "loss": manifest.get("training", {}).get("final_loss", 0),
        "eval_loss": 0,  # Added if eval is run
        "trained_at": manifest.get("completed_at", ""),
    }

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{ledger_url}/api/admin/model",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Admin-Key": admin_key,
            "User-Agent": "SwarmJelly/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        print(f"  [ledger] HTTP {e.code}: {err_body}")
        return None
    except Exception as e:
        print(f"  [ledger] Push failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description=f"Train {BUILD_NAME}")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--resume", type=str)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer, EarlyStoppingCallback
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    import torch

    print("=" * 70)
    print(f"  {BUILD_NAME} — SELF-HEALING INTELLIGENCE")
    print(f"  Base:       {MODEL_NAME}")
    print(f"  Mission:    Learn failure patterns → generate Royal Jelly")
    print(f"  Taxonomy:   7 failure modes (reasoning/knowledge/instruction/agent)")
    print(f"  Tasks:      DIAGNOSE, REPAIR, PREVENT, DETECT, COMPARE")
    print(f"  Method:     bf16 LoRA r={LORA_R} alpha={LORA_ALPHA}")
    print(f"  LR:         {LEARNING_RATE}")
    print(f"  Batch:      {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print(f"  Thinking:   DISABLED")
    if args.smoke_test:
        print(f"  Mode:       SMOKE TEST")
    elif args.pilot:
        print(f"  Mode:       PILOT")
    else:
        print(f"  Mode:       FULL")
    print("=" * 70)

    print("\n[0/5] Validating data...")
    data_info = validate_data(TRAIN_FILE, EVAL_FILE)

    print("\n[1/5] Loading model (4B bf16, ~8GB)...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    print("[2/5] Loading tokenizer (AutoTokenizer bypass)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"

    print("[3/5] Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    model.print_trainable_parameters()

    print("[4/5] Loading dataset...")
    train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    eval_dataset = load_dataset("json", data_files=EVAL_FILE, split="train")

    if args.smoke_test:
        train_dataset = train_dataset.select(range(min(500, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(50, len(eval_dataset))))
    elif args.pilot:
        train_dataset = train_dataset.select(range(min(5000, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(200, len(eval_dataset))))

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"text": text}

    train_dataset = train_dataset.map(
        format_chat,
        remove_columns=train_dataset.column_names,
        desc="Formatting train",
        num_proc=16,
    )
    eval_dataset = eval_dataset.map(
        format_chat,
        remove_columns=eval_dataset.column_names,
        desc="Formatting eval",
        num_proc=4,
    )

    if len(eval_dataset) > MAX_EVAL_SAMPLES:
        eval_dataset = eval_dataset.select(range(MAX_EVAL_SAMPLES))

    eff_batch = BATCH_SIZE * GRAD_ACCUM
    full_epoch_steps = len(train_dataset) // eff_batch
    max_steps = int(full_epoch_steps * MAX_EPOCH_FRACTION)

    print(f"  Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")
    print(f"  Eff batch:     {eff_batch}")
    print(f"  Full epoch:    {full_epoch_steps} steps")
    print(f"  Max steps:     {max_steps} (capped at {MAX_EPOCH_FRACTION} epoch)")

    print("[5/5] Configuring trainer...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            max_steps=max_steps,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=LR_SCHEDULER,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            max_seq_length=MAX_SEQ_LEN,
            packing=True,
            dataset_text_field="text",
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            ),
        ],
    )

    print("\n" + "=" * 70)
    print(f"  TRAINING START — {BUILD_NAME}")
    print(f"  LR={LEARNING_RATE}, batch={eff_batch}, scheduler={LR_SCHEDULER}")
    print(f"  7 failure modes × 5 Royal Jelly tasks = self-healing intelligence")
    print("=" * 70)

    if args.resume:
        result = trainer.train(resume_from_checkpoint=args.resume)
    else:
        result = trainer.train()

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE — {BUILD_NAME}")
    print(f"  Loss:    {result.training_loss:.4f}")
    print(f"  Steps:   {result.global_step}")
    print(f"  Time:    {elapsed/3600:.2f}h ({elapsed/60:.0f}m)")
    print("=" * 70)

    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"  Adapter saved to: {OUTPUT_DIR}")

    print("\n  Merging adapter into base model...")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(MERGED_DIR),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"  Merged model saved to: {MERGED_DIR}")

    manifest = {
        "model": BUILD_NAME,
        "base": MODEL_NAME,
        "method": f"bf16 LoRA r={LORA_R} alpha={LORA_ALPHA}",
        "config_source": "Swarm Gold Standard (4B tier)",
        "mission": "Self-healing intelligence — failure analysis, repair generation, failure taxonomy classification",
        "failure_taxonomy": {
            "reasoning": ["missing_step", "false_assumption"],
            "knowledge": ["hallucination", "overgeneralization"],
            "instruction": ["drift", "schema_break"],
            "agent": ["tool_misuse"],
        },
        "royal_jelly_tasks": ["DIAGNOSE", "REPAIR", "PREVENT", "DETECT", "COMPARE"],
        "data": {
            **data_info,
            "composition": {
                "selfheal": "~45K (Royal Jelly — repair strategies)",
                "failure": "~21K (failure intelligence — diagnosis)",
                "behavior": "~39K (agent workflows — behavioral patterns)",
                "judge_traces": "~164K (quality evaluation trajectories)",
            },
        },
        "training": {
            "steps": result.global_step,
            "max_steps": max_steps,
            "final_loss": round(result.training_loss, 4),
            "learning_rate": LEARNING_RATE,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM,
            "max_seq_len": MAX_SEQ_LEN,
            "packing": True,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "epoch_fraction": MAX_EPOCH_FRACTION,
        },
        "hardware": {
            "gpu": "NVIDIA RTX PRO 6000 Blackwell",
            "vram_gb": 96,
        },
        "deployment_targets": [
            "Jetson Orin Nano 8GB (Q4_K_M GGUF)",
            "swarmrails CPU (bf16/Q8_0)",
            "Any 8GB+ GPU (bf16)",
        ],
        "elapsed_hours": round(elapsed / 3600, 2),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "adapter_path": str(OUTPUT_DIR),
        "merged_path": str(MERGED_DIR),
    }

    manifest_path = Path("/data2/swarmjelly-4b/MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")

    # ── Push model lineage to hive-ledger ──
    push_result = push_model_lineage(manifest, data_info)
    if push_result:
        print(f"  Ledger: {push_result}")

    print(f"\n  Next steps:")
    print(f"  1. Fix vision config if needed")
    print(f"  2. Quantize: llama-quantize {MERGED_DIR} swarmjelly-4b-q4_k_m.gguf Q4_K_M")
    print(f"  3. Deploy on Jetson edge or swarmrails CPU")
    print(f"  4. Wire into PropolisCollector → SwarmJelly → vet pipeline → ledger")
    print("=" * 70)


if __name__ == "__main__":
    main()
