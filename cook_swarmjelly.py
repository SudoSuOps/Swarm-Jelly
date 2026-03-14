#!/usr/bin/env python3
"""SwarmJelly-4B Dataset Assembly
=================================
Combines Royal Jelly (selfheal + failure + behavior) + judge traces
into a unified messages-format training dataset.

Sources:
  - selfheal_shard_000-009: 45,417 pairs (IRO format)
  - nuked_failure_shard_000-004: 20,829 pairs (IRO format)
  - nuked_behavior_shard_000-007: 38,936 pairs (IRO format)
  - judge traces (4 files, messages format, post-vet clean)

Usage:
  python3 cook_swarmjelly.py --dry-run
  python3 cook_swarmjelly.py
  python3 cook_swarmjelly.py --include-judge /data2/judge_backfill/
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

NUKED_DIR = Path("/data2/nuked_backfill")
JUDGE_DIR = Path("/data2/judge_backfill")
OUTPUT_DIR = Path("/data2/swarmjelly-4b")
EVAL_RATIO = 0.10
SEED = 42

SWARMJELLY_SYSTEM = (
    "You are SwarmJelly — the self-healing intelligence engine for the SwarmHive. "
    "You analyze agent failures, design repair strategies, classify failure modes, "
    "and generate quality guardrails. You transform propolis (failures) into Royal Jelly "
    "(self-healing training pairs). You understand agent execution trajectories, "
    "failure root causes, and recovery patterns across all domains."
)


def iro_to_messages(rec: dict, source_type: str) -> dict | None:
    """Convert IRO (instruction/reasoning/output) to messages format."""
    instruction = rec.get("instruction", "").strip()
    reasoning = rec.get("reasoning", "").strip()
    output = rec.get("output", "").strip()

    if not instruction or not output:
        return None

    # Build assistant response
    if reasoning:
        assistant = f"{reasoning}\n\n{output}"
    else:
        assistant = output

    messages = [
        {"role": "system", "content": SWARMJELLY_SYSTEM},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": assistant},
    ]

    return {
        "messages": messages,
        "metadata": {
            "source": source_type,
            "vertical": rec.get("vertical", ""),
            "score": rec.get("score", 0),
            "mechanism": rec.get("mechanism", ""),
            "fingerprint": rec.get("fingerprint", ""),
        },
    }


def load_iro_shards(pattern: str, source_type: str) -> list[dict]:
    """Load IRO-format shards and convert to messages."""
    pairs = []
    files = sorted(NUKED_DIR.glob(pattern))
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    converted = iro_to_messages(rec, source_type)
                    if converted:
                        pairs.append(converted)
                except json.JSONDecodeError:
                    continue
    return pairs


FAILURE_KEYWORDS = {
    "hallucination": ["hallucin"],
    "schema_break": ["schema", "format violation", "json valid"],
    "missing_step": ["missing", "reasoning", "wrong_reasoning", "incomplete"],
    "tool_misuse": ["tool", "retrieval", "function call"],
    "instruction_drift": ["drift", "instruction", "ignored"],
    "overgeneralization": ["general", "broad", "universal"],
    "safety_bypass": ["safety", "bypass", "pii", "leak"],
    "cascading_error": ["cascading", "stale"],
    "clean_pass": ["clean_pass"],
}


def classify_failure_type(user_content: str) -> str:
    """Classify a judge trace by failure type from user content keywords."""
    ul = user_content.lower()
    for ftype, keywords in FAILURE_KEYWORDS.items():
        for kw in keywords:
            if kw in ul:
                return ftype
    return "other"


def load_judge_traces(judge_dir: Path, cap: int = 0) -> list[dict]:
    """Load judge traces with optional stratified cap for balance.

    If cap > 0, stratified sample by failure type to ensure balanced distribution.
    """
    # First pass: load all and classify
    by_type: dict[str, list[dict]] = {}
    total_loaded = 0

    files = sorted(judge_dir.glob("*.jsonl"))
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    msgs = rec.get("messages", [])
                    if len(msgs) < 2:
                        continue

                    total_len = sum(len(m.get("content", "")) for m in msgs)
                    if total_len < 100:
                        continue

                    user_content = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                    ftype = classify_failure_type(user_content)

                    pair = {
                        "messages": msgs,
                        "metadata": {
                            "source": "judge_trace",
                            "vertical": rec.get("metadata", {}).get("vertical", "judge"),
                            "failure_type": ftype,
                        },
                    }
                    by_type.setdefault(ftype, []).append(pair)
                    total_loaded += 1
                except json.JSONDecodeError:
                    continue

    print(f"    {total_loaded:,} judge traces loaded across {len(by_type)} failure types")
    print(f"    Raw distribution:")
    for ftype in sorted(by_type, key=lambda k: -len(by_type[k])):
        pct = len(by_type[ftype]) / total_loaded * 100
        print(f"      {ftype:25s} {len(by_type[ftype]):8,} ({pct:.1f}%)")

    if cap <= 0 or total_loaded <= cap:
        # No capping needed
        pairs = []
        for bucket in by_type.values():
            pairs.extend(bucket)
        return pairs

    # Stratified sampling: equal share per type, with remainder to smallest types
    n_types = len(by_type)
    per_type = cap // n_types
    remainder = cap - (per_type * n_types)

    pairs = []
    # Sort by size ascending so smaller types get full representation
    sorted_types = sorted(by_type.items(), key=lambda x: len(x[1]))

    print(f"\n    Stratified cap: {cap:,} total, {per_type:,} per type ({n_types} types)")

    for i, (ftype, bucket) in enumerate(sorted_types):
        # Smaller types might have fewer than per_type — take all, redistribute remainder
        take = min(len(bucket), per_type + (1 if i < remainder else 0))
        random.shuffle(bucket)
        sampled = bucket[:take]
        pairs.extend(sampled)
        print(f"      {ftype:25s} {take:6,} / {len(bucket):6,}")

    print(f"    Total sampled: {len(pairs):,}")
    return pairs


def dedup_pairs(pairs: list[dict]) -> list[dict]:
    """Dedup by MD5 of assistant content."""
    seen = set()
    clean = []
    for p in pairs:
        msgs = p["messages"]
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        fp = hashlib.md5(assistant.strip().lower().encode()).hexdigest()
        if fp not in seen:
            seen.add(fp)
            clean.append(p)
    return clean


def main():
    parser = argparse.ArgumentParser(description="Assemble SwarmJelly-4B training dataset")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-judge", type=str, help="Path to judge traces dir")
    parser.add_argument("--judge-cap", type=int, default=50000, help="Max judge traces (stratified sample)")
    parser.add_argument("--max-total", type=int, default=0, help="Max total pairs after dedup (0=unlimited)")
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("=" * 70)
    print("  SWARMJELLY-4B DATASET ASSEMBLY")
    print("=" * 70)

    # Load selfheal
    print("\n  Loading selfheal shards...")
    selfheal = load_iro_shards("selfheal_shard_*.jsonl", "selfheal")
    print(f"    {len(selfheal):,} selfheal pairs loaded")

    # Load failure
    print("  Loading failure shards...")
    failure = load_iro_shards("nuked_failure_shard_*.jsonl", "failure")
    print(f"    {len(failure):,} failure pairs loaded")

    # Load behavior
    print("  Loading behavior shards...")
    behavior = load_iro_shards("nuked_behavior_shard_*.jsonl", "behavior")
    print(f"    {len(behavior):,} behavior pairs loaded")

    # Load judge traces (optional)
    judge = []
    if args.include_judge:
        judge_path = Path(args.include_judge)
        if judge_path.exists():
            print(f"  Loading judge traces from {judge_path} (cap={args.judge_cap:,})...")
            judge = load_judge_traces(judge_path, cap=args.judge_cap)
            print(f"    {len(judge):,} judge traces after stratified sampling")
        else:
            print(f"    WARNING: Judge dir not found: {judge_path}")

    # Combine
    all_pairs = selfheal + failure + behavior + judge
    print(f"\n  Total raw: {len(all_pairs):,}")

    # Source distribution
    sources = Counter(p["metadata"]["source"] for p in all_pairs)
    print("  Source distribution:")
    for src, count in sources.most_common():
        print(f"    {src:20s} {count:8,}")

    # Dedup
    print("\n  Deduplicating...")
    clean = dedup_pairs(all_pairs)
    dupes = len(all_pairs) - len(clean)
    print(f"    Removed {dupes:,} duplicates ({dupes/len(all_pairs)*100:.1f}%)")
    print(f"    Clean:  {len(clean):,}")

    # Total cap (stratified by source to maintain balance)
    if args.max_total > 0 and len(clean) > args.max_total:
        random.seed(args.seed)
        random.shuffle(clean)
        clean = clean[:args.max_total]
        print(f"    Capped to {len(clean):,} total pairs")

        # Recount source distribution after cap
        sources = Counter(p["metadata"]["source"] for p in clean)
        print("  Source distribution (post-cap):")
        for src, count in sources.most_common():
            print(f"    {src:20s} {count:8,}")

    if args.dry_run:
        print(f"\n  DRY RUN — would write {len(clean):,} pairs")
        return

    # Shuffle
    random.seed(args.seed)
    random.shuffle(clean)

    # Split
    eval_count = int(len(clean) * args.eval_ratio)
    train_data = clean[eval_count:]
    eval_data = clean[:eval_count]

    print(f"\n  Train: {len(train_data):,}")
    print(f"  Eval:  {len(eval_data):,}")

    # Write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "swarmjelly_train.jsonl"
    eval_path = OUTPUT_DIR / "swarmjelly_eval.jsonl"

    for path, data, label in [
        (train_path, train_data, "Train"),
        (eval_path, eval_data, "Eval"),
    ]:
        with open(path, "w") as f:
            for rec in data:
                f.write(json.dumps({"messages": rec["messages"]}, separators=(",", ":")) + "\n")
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"  {label}: {path} ({len(data):,} records, SHA256: {sha[:16]}...)")

    # Source distribution in train set
    train_sources = Counter(p["metadata"]["source"] for p in train_data)
    print(f"\n  Train source mix:")
    for src, count in train_sources.most_common():
        pct = count / len(train_data) * 100
        print(f"    {src:20s} {count:8,} ({pct:.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"  ASSEMBLY COMPLETE")
    print(f"  Train: {train_path}")
    print(f"  Eval:  {eval_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
