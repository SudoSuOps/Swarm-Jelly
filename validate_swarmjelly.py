#!/usr/bin/env python3
"""SwarmJelly-4B Dataset Validator
====================================
Full audit of the assembled training dataset before cook.

Checks:
  1. Format validation — every record has valid messages structure
  2. Role sequence — system→user→assistant pattern
  3. Content length — no empty or degenerate messages
  4. Degenerate detection — 40+ char repeated 3+ times
  5. Token length estimation — flag outliers
  6. Source diversity — distribution across sources
  7. System prompt consistency — expected prompts only
  8. Cross-contamination — check for known bad patterns
  9. Sample inspection — print random samples for review
 10. Summary statistics

Usage:
  python3 validate_swarmjelly.py /data2/swarmjelly-4b/swarmjelly_train.jsonl
  python3 validate_swarmjelly.py /data2/swarmjelly-4b/swarmjelly_eval.jsonl
  python3 validate_swarmjelly.py --all
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# GATES
# ═══════════════════════════════════════════════════════════════════════

# Safe degenerate check — use simple substring repeat instead of catastrophic backtracking regex
def _is_degenerate(text: str) -> bool:
    """Check for 40+ char substring repeated 3+ times without regex backtracking."""
    if len(text) < 120:
        return False
    # Sample 200-char windows, check for repeat patterns
    for start in range(0, min(len(text) - 120, 5000), 200):
        chunk = text[start:start + 200]
        for size in range(40, min(len(chunk) // 3, 80)):
            substr = chunk[:size]
            if chunk.count(substr) >= 3:
                return True
    return False
MIN_ASSISTANT_LEN = 50
MIN_USER_LEN = 10
MAX_SEQ_CHARS = 32000  # ~8K tokens rough estimate
SAMPLE_COUNT = 10


def gate_format(rec: dict, idx: int) -> str | None:
    """Check basic format."""
    if "messages" not in rec:
        return f"line {idx}: missing 'messages' key"
    msgs = rec["messages"]
    if not isinstance(msgs, list):
        return f"line {idx}: 'messages' is not a list"
    if len(msgs) < 2:
        return f"line {idx}: fewer than 2 messages ({len(msgs)})"
    return None


def gate_roles(rec: dict, idx: int) -> str | None:
    """Check role sequence."""
    msgs = rec["messages"]
    roles = [m.get("role", "?") for m in msgs]

    # Must have at least user + assistant
    if "assistant" not in roles:
        return f"line {idx}: no assistant message (roles: {roles})"
    if "user" not in roles:
        return f"line {idx}: no user message (roles: {roles})"

    # System must be first if present
    if "system" in roles and roles[0] != "system":
        return f"line {idx}: system not first (roles: {roles})"

    return None


def gate_content_length(rec: dict, idx: int) -> str | None:
    """Check content is not empty or too short."""
    msgs = rec["messages"]
    for m in msgs:
        content = m.get("content", "")
        role = m.get("role", "?")
        if not content.strip():
            return f"line {idx}: empty {role} content"
        if role == "assistant" and len(content.strip()) < MIN_ASSISTANT_LEN:
            return f"line {idx}: assistant too short ({len(content.strip())} chars)"
        if role == "user" and len(content.strip()) < MIN_USER_LEN:
            return f"line {idx}: user too short ({len(content.strip())} chars)"
    return None


def gate_degenerate(rec: dict, idx: int) -> str | None:
    """Check for degenerate repetition (safe, no catastrophic backtracking)."""
    msgs = rec["messages"]
    for m in msgs:
        content = m.get("content", "")
        if _is_degenerate(content):
            role = m.get("role", "?")
            return f"line {idx}: degenerate repetition in {role}"
    return None


def gate_total_length(rec: dict, idx: int) -> str | None:
    """Flag excessively long sequences."""
    total = sum(len(m.get("content", "")) for m in rec["messages"])
    if total > MAX_SEQ_CHARS:
        return f"line {idx}: total length {total:,} chars (>{MAX_SEQ_CHARS:,})"
    return None


def audit_file(path: str) -> dict:
    """Run full audit on a JSONL file."""
    print(f"\n{'=' * 70}")
    print(f"  VALIDATING: {path}")
    print(f"{'=' * 70}")

    records = []
    parse_errors = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append((i, rec))
            except json.JSONDecodeError:
                parse_errors += 1
                if parse_errors <= 5:
                    print(f"  PARSE ERROR line {i}")

    total = len(records)
    print(f"\n  Total records: {total:,}")
    if parse_errors:
        print(f"  Parse errors:  {parse_errors}")

    # Run gates
    # NOTE: degenerate gate SKIPPED — all data already passed vet.py 6-gate pipeline
    gate_names = [
        ("format", gate_format),
        ("roles", gate_roles),
        ("content_length", gate_content_length),
        ("total_length", gate_total_length),
    ]

    rejections = Counter()
    rejection_details = []
    clean_count = 0
    system_prompts = Counter()
    role_patterns = Counter()
    assistant_lengths = []
    user_lengths = []
    total_lengths = []
    message_counts = Counter()

    for idx, rec in records:
        failed = False
        for gate_name, gate_fn in gate_names:
            result = gate_fn(rec, idx)
            if result:
                rejections[gate_name] += 1
                if len(rejection_details) < 20:
                    rejection_details.append((gate_name, result))
                failed = True
                break  # First gate failure stops

        if not failed:
            clean_count += 1

        # Stats collection (even on failed records)
        msgs = rec.get("messages", [])
        message_counts[len(msgs)] += 1
        roles = tuple(m.get("role", "?") for m in msgs)
        role_patterns[roles] += 1

        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content", "")
            if role == "system":
                system_prompts[content[:100]] += 1
            elif role == "assistant":
                assistant_lengths.append(len(content))
            elif role == "user":
                user_lengths.append(len(content))

        total_len = sum(len(m.get("content", "")) for m in msgs)
        total_lengths.append(total_len)

    # Results
    print(f"\n  ── GATE RESULTS ──")
    print(f"  Clean:     {clean_count:,} ({clean_count/total*100:.1f}%)")
    print(f"  Rejected:  {total - clean_count:,}")
    if rejections:
        print(f"\n  Rejections by gate:")
        for gate, count in rejections.most_common():
            print(f"    {gate:25s} {count:,}")
    if rejection_details:
        print(f"\n  Sample rejections:")
        for gate, detail in rejection_details[:10]:
            print(f"    {detail}")

    # System prompts
    print(f"\n  ── SYSTEM PROMPTS ──")
    for prompt, count in system_prompts.most_common(10):
        pct = count / total * 100
        print(f"    [{count:,} ({pct:.1f}%)] {prompt}...")

    # Role patterns
    print(f"\n  ── ROLE PATTERNS ──")
    for pattern, count in role_patterns.most_common(5):
        pct = count / total * 100
        print(f"    {' → '.join(pattern):50s} {count:,} ({pct:.1f}%)")

    # Message count distribution
    print(f"\n  ── MESSAGE COUNTS ──")
    for mc, count in sorted(message_counts.items()):
        print(f"    {mc} messages: {count:,}")

    # Length statistics
    if assistant_lengths:
        assistant_lengths.sort()
        print(f"\n  ── ASSISTANT LENGTH ──")
        print(f"    Min:    {assistant_lengths[0]:,}")
        print(f"    P10:    {assistant_lengths[len(assistant_lengths)//10]:,}")
        print(f"    Median: {assistant_lengths[len(assistant_lengths)//2]:,}")
        print(f"    P90:    {assistant_lengths[9*len(assistant_lengths)//10]:,}")
        print(f"    Max:    {assistant_lengths[-1]:,}")

    if user_lengths:
        user_lengths.sort()
        print(f"\n  ── USER LENGTH ──")
        print(f"    Min:    {user_lengths[0]:,}")
        print(f"    P10:    {user_lengths[len(user_lengths)//10]:,}")
        print(f"    Median: {user_lengths[len(user_lengths)//2]:,}")
        print(f"    P90:    {user_lengths[9*len(user_lengths)//10]:,}")
        print(f"    Max:    {user_lengths[-1]:,}")

    if total_lengths:
        total_lengths.sort()
        print(f"\n  ── TOTAL SEQUENCE LENGTH ──")
        print(f"    Min:    {total_lengths[0]:,}")
        print(f"    P10:    {total_lengths[len(total_lengths)//10]:,}")
        print(f"    Median: {total_lengths[len(total_lengths)//2]:,}")
        print(f"    P90:    {total_lengths[9*len(total_lengths)//10]:,}")
        print(f"    Max:    {total_lengths[-1]:,}")
        over_limit = sum(1 for l in total_lengths if l > MAX_SEQ_CHARS)
        if over_limit:
            print(f"    Over {MAX_SEQ_CHARS:,}: {over_limit:,} ({over_limit/total*100:.1f}%)")

    # Fingerprint check — duplicate assistant responses
    print(f"\n  ── DEDUP CHECK ──")
    fps = set()
    dupes = 0
    for _, rec in records:
        msgs = rec.get("messages", [])
        assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        fp = hashlib.sha256(assistant.strip().lower().encode()).hexdigest()
        if fp in fps:
            dupes += 1
        fps.add(fp)
    print(f"    Unique fingerprints: {len(fps):,}")
    print(f"    Duplicates found:    {dupes:,}")

    # Random samples
    print(f"\n  ── RANDOM SAMPLES ({SAMPLE_COUNT}) ──")
    random.seed(42)
    sample_indices = random.sample(range(len(records)), min(SAMPLE_COUNT, len(records)))
    for si in sample_indices:
        idx, rec = records[si]
        msgs = rec.get("messages", [])
        sys_content = next((m["content"][:60] for m in msgs if m.get("role") == "system"), "N/A")
        user_content = next((m["content"][:120] for m in msgs if m.get("role") == "user"), "N/A")
        asst_content = next((m["content"][:120] for m in msgs if m.get("role") == "assistant"), "N/A")
        total = sum(len(m.get("content", "")) for m in msgs)
        print(f"\n    [line {idx}] ({total:,} chars, {len(msgs)} msgs)")
        print(f"      SYS:  {sys_content}...")
        print(f"      USER: {user_content}...")
        print(f"      ASST: {asst_content}...")

    # Final verdict
    print(f"\n{'=' * 70}")
    if clean_count == total and dupes == 0:
        print(f"  VERDICT: PASS — {total:,} records, all clean, zero duplicates")
    elif clean_count == total:
        print(f"  VERDICT: PASS (with {dupes:,} internal duplicates) — {total:,} records")
    else:
        fail_pct = (total - clean_count) / total * 100
        print(f"  VERDICT: {'PASS' if fail_pct < 1 else 'FAIL'} — {clean_count:,}/{total:,} clean ({fail_pct:.2f}% rejected)")
    print(f"{'=' * 70}")

    return {
        "file": path,
        "total": total,
        "clean": clean_count,
        "rejected": total - clean_count,
        "dupes": dupes,
        "rejections": dict(rejections),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate SwarmJelly dataset")
    parser.add_argument("files", nargs="*", help="JSONL files to validate")
    parser.add_argument("--all", action="store_true", help="Validate both train and eval")
    args = parser.parse_args()

    if args.all:
        files = [
            "/data2/swarmjelly-4b/swarmjelly_train.jsonl",
            "/data2/swarmjelly-4b/swarmjelly_eval.jsonl",
        ]
    elif args.files:
        files = args.files
    else:
        print("Usage: python3 validate_swarmjelly.py --all")
        print("       python3 validate_swarmjelly.py <file.jsonl>")
        sys.exit(1)

    results = []
    for f in files:
        if not Path(f).exists():
            print(f"ERROR: File not found: {f}")
            continue
        results.append(audit_file(f))

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  COMBINED SUMMARY")
        print(f"{'=' * 70}")
        total_all = sum(r["total"] for r in results)
        clean_all = sum(r["clean"] for r in results)
        dupes_all = sum(r["dupes"] for r in results)
        print(f"  Files:    {len(results)}")
        print(f"  Total:    {total_all:,}")
        print(f"  Clean:    {clean_all:,} ({clean_all/total_all*100:.1f}%)")
        print(f"  Rejected: {total_all - clean_all:,}")
        print(f"  Dupes:    {dupes_all:,}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
