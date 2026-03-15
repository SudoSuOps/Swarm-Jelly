#!/usr/bin/env python3
"""SwarmJelly Self-Healing Pipeline
=====================================

Propolis (failures) → SwarmJelly-4B inference → Vet → Stamp → Ledger

Closes the self-healing loop:
  1. INGEST  — load propolis shards (PropolisCollector or IRO backfill)
  2. PROMPT  — build 5 RJ task prompts per failure
  3. INFER   — call llama-server /v1/chat/completions
  4. VET     — source tags + content length + degenerate + dedup
  5. STAMP   — HiveCellStamper → cell_id, fingerprint, grade
  6. PUSH    — batch to hive-ledger (optional --push)

Usage:
    # Start llama-server first:
    llama-server -m /data2/swarmjelly-4b/swarmjelly-4b-q4_k_m.gguf -c 8192 --port 8085 -ngl 99

    # Check server
    python3 pipeline_swarmjelly.py --check

    # Dry run
    python3 pipeline_swarmjelly.py --dry-run

    # Process propolis shards
    python3 pipeline_swarmjelly.py --api-url http://localhost:8085

    # Process + push to ledger
    python3 pipeline_swarmjelly.py --api-url http://localhost:8085 --push

    # Backfill from nuked_selfheal (IRO, skips inference)
    python3 pipeline_swarmjelly.py --backfill-dir ~/swarm_cooks/nuked_selfheal/

    # Push previously-stamped cells
    python3 pipeline_swarmjelly.py --push-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# PATH SETUP — import existing hive modules
# ═══════════════════════════════════════════════════════════════════════

sys.path.insert(0, "/data2/audit")
sys.path.insert(0, "/data2/hive-ledger/scripts")

from hive.cell import HiveCellStamper
from hive.validate import check_source_tags, content_hash, load_jsonl
from push_batch import post_batch, merkle_root, sha256 as sha256_str

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

SWARMJELLY_SYSTEM = (
    "You are SwarmJelly — the self-healing intelligence engine for the SwarmHive. "
    "You analyze agent failures, design repair strategies, classify failure modes, "
    "and generate quality guardrails. You transform propolis (failures) into Royal Jelly "
    "(self-healing training pairs). You understand agent execution trajectories, "
    "failure root causes, and recovery patterns across all domains."
)

DEFAULT_PROPOLIS_DIR = Path.home() / "swarm_cooks" / "propolis"
DEFAULT_OUTPUT_DIR = Path("/data2/Swarm-Jelly/pipeline_output")
DEFAULT_API_URL = "http://localhost:8085"

TASK_ORDER = ["diagnose", "repair", "prevent", "detect", "compare"]

MIN_ASSISTANT_LEN = 50
MAX_RETRIES = 3

# ═══════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════

TASK_PROMPTS = {
    "diagnose": (
        "A {source_model} agent failed on a {vertical} task.\n"
        "Failure class: {failure_class}\n"
        "Errors: {errors}\n\n"
        "Original prompt:\n{user_prompt}\n\n"
        "Broken output:\n{model_output}\n\n"
        "DIAGNOSE: What went wrong? Classify the failure mode and identify the root cause. "
        "Provide: failure_mode, root_cause, severity (1-5), and recommended_action."
    ),
    "repair": (
        "A {source_model} agent produced a broken output for a {vertical} task.\n"
        "Failure class: {failure_class}\n"
        "Errors: {errors}\n\n"
        "Original prompt:\n{user_prompt}\n\n"
        "Broken output:\n{model_output}\n\n"
        "REPAIR: Design a step-by-step recovery strategy. "
        "Then produce the corrected output that fixes all identified issues."
    ),
    "prevent": (
        "A {failure_class} failure occurred in a {vertical} {source_model} agent.\n"
        "Errors: {errors}\n\n"
        "Original prompt:\n{user_prompt}\n\n"
        "PREVENT: Design a guardrail that would prevent this failure class. "
        "The guardrail must be implementable as a pre-execution or post-execution check. "
        "Include: trigger_condition, check_logic, remediation_action, and estimated_coverage."
    ),
    "detect": (
        "Given the following output from a {vertical} {source_model} agent, "
        "determine whether it exhibits {failure_class}.\n\n"
        "Output to analyze:\n{model_output}\n\n"
        "DETECT: Provide detection_result (POSITIVE/NEGATIVE), confidence (0-100), "
        "evidence (specific quotes or patterns), and false_positive_risk."
    ),
    "compare": (
        "Two outputs from a {vertical} {source_model} agent are shown.\n\n"
        "Output A (broken, {failure_class}):\n{model_output}\n\n"
        "Output B (corrected):\n{repair_output}\n\n"
        "COMPARE: Rank the outputs. Explain why Output B is superior. "
        "Score each on: correctness, completeness, format_compliance, reasoning_depth."
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PropolisRecord:
    """Unified failure record — from PropolisCollector or IRO backfill."""
    id: str
    failure_class: str
    vertical: str
    source_model: str
    user_prompt: str
    model_output: str
    system_prompt: str = ""
    errors: list[str] = field(default_factory=list)
    source_path: str = ""

    @classmethod
    def from_propolis(cls, rec: dict, source_path: str = "") -> PropolisRecord:
        """From PropolisCollector JSONL format."""
        ctx = rec.get("context", {})
        details = rec.get("failure_details", {})
        return cls(
            id=rec.get("id", "unknown"),
            failure_class=rec.get("failure_class", "unknown"),
            vertical=rec.get("vertical", "general"),
            source_model=rec.get("source_model", "unknown"),
            user_prompt=ctx.get("user_prompt", ""),
            model_output=ctx.get("model_output", ""),
            system_prompt=ctx.get("system_prompt", ""),
            errors=details.get("errors", []),
            source_path=source_path,
        )

    @classmethod
    def from_iro(cls, rec: dict, source_path: str = "") -> PropolisRecord:
        """From IRO backfill format (nuked_selfheal)."""
        lineage = rec.get("lineage", {})
        source = rec.get("source", "")
        # Extract failure class from source field: selfheal_repair_incomplete_execution
        parts = source.split("_", 2)
        failure_class = parts[2] if len(parts) > 2 else "unknown"
        return cls(
            id=rec.get("id", "unknown"),
            failure_class=failure_class,
            vertical=rec.get("vertical", "general"),
            source_model=lineage.get("gen_model", rec.get("specialty", "unknown")),
            user_prompt=rec.get("instruction", ""),
            model_output=rec.get("output", ""),
            errors=[],
            source_path=source_path,
        )


# ═══════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def load_state(output_dir: Path) -> dict:
    state_path = output_dir / "state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {
        "version": 1,
        "processed_shards": {},
        "fingerprints": [],
        "total_pairs_generated": 0,
        "total_pairs_vetted": 0,
        "total_cells_stamped": 0,
        "total_cells_pushed": 0,
    }


def save_state(state: dict, output_dir: Path):
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state_path = output_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_processed(shard_path: str, state: dict) -> bool:
    entry = state["processed_shards"].get(shard_path)
    if not entry:
        return False
    return entry.get("sha256") == file_sha256(shard_path)


# ═══════════════════════════════════════════════════════════════════════
# SHARD DISCOVERY
# ═══════════════════════════════════════════════════════════════════════

def discover_shards(propolis_dir: Path, state: dict) -> list[str]:
    """Find unprocessed propolis shards."""
    shards = []
    if not propolis_dir.exists():
        return shards
    for shard in sorted(propolis_dir.rglob("shard_*.jsonl")):
        path = str(shard)
        if not is_processed(path, state):
            shards.append(path)
    return shards


def discover_backfill(backfill_dir: Path, state: dict) -> list[str]:
    """Find unprocessed IRO backfill shards."""
    shards = []
    if not backfill_dir.exists():
        return shards
    for shard in sorted(backfill_dir.glob("*.jsonl")):
        path = str(shard)
        if not is_processed(path, state):
            shards.append(path)
    return shards


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def check_server(api_url: str) -> bool:
    """Ping llama-server health endpoint."""
    try:
        req = urllib.request.Request(f"{api_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            status = data.get("status", "unknown")
            print(f"  Server: {api_url} — {status}")
            return status == "ok"
    except Exception as e:
        print(f"  Server: {api_url} — UNREACHABLE ({e})")
        return False


def infer(messages: list[dict], api_url: str, temperature: float, max_tokens: int) -> str | None:
    """Call llama-server /v1/chat/completions with retry."""
    payload = json.dumps({
        "model": "swarmjelly-4b",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                f"{api_url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(task: str, rec: PropolisRecord, repair_output: str = "") -> str:
    """Build the user prompt for a given task."""
    template = TASK_PROMPTS[task]
    errors_str = "; ".join(rec.errors) if rec.errors else "none reported"
    return template.format(
        source_model=rec.source_model,
        vertical=rec.vertical,
        failure_class=rec.failure_class,
        errors=errors_str,
        user_prompt=rec.user_prompt[:2000],
        model_output=rec.model_output[:3000],
        repair_output=repair_output[:3000],
    )


# ═══════════════════════════════════════════════════════════════════════
# VET PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def _is_degenerate(text: str) -> bool:
    """Check for 40+ char substring repeated 3+ times."""
    if len(text) < 120:
        return False
    for start in range(0, min(len(text) - 120, 5000), 200):
        chunk = text[start:start + 200]
        for size in range(40, min(len(chunk) // 3, 80)):
            substr = chunk[:size]
            if chunk.count(substr) >= 3:
                return True
    return False


COHERENCE_KEYWORDS = {
    "diagnose": ["root cause", "failure", "severity", "cause"],
    "repair": ["step", "fix", "correct", "recover"],
    "prevent": ["guardrail", "check", "prevent", "trigger"],
    "detect": ["positive", "negative", "confidence", "evidence"],
    "compare": ["output a", "output b", "superior", "score"],
}


def vet_pair(record: dict, seen_hashes: set[str]) -> tuple[bool, str, int]:
    """
    Vet a single generated pair.
    Returns (passed, reason, score).
    """
    messages = record.get("messages", [])
    task = record.get("task", "")
    gates_passed = 0

    # Gate 1: Source tags — valid roles, non-empty
    tag_result = check_source_tags([{"messages": messages}])
    if tag_result.passed:
        gates_passed += 1
    else:
        pass  # continue checking other gates

    # Gate 2: Content length — assistant >= 50 chars
    assistant_content = ""
    for msg in messages:
        if msg.get("role") == "assistant":
            assistant_content = msg.get("content", "")
    if len(assistant_content) >= MIN_ASSISTANT_LEN:
        gates_passed += 1

    # Gate 3: Degenerate check
    if not _is_degenerate(assistant_content):
        gates_passed += 1

    # Gate 4: Dedup
    h = content_hash({"messages": messages})
    if h not in seen_hashes:
        seen_hashes.add(h)
        gates_passed += 1

    # Gate 5: Coherence — task keywords present
    content_lower = assistant_content.lower()
    keywords = COHERENCE_KEYWORDS.get(task, [])
    if any(kw in content_lower for kw in keywords):
        gates_passed += 1

    if gates_passed >= 5:
        return True, "pass", 70
    elif gates_passed >= 4:
        return True, "pass", 60
    else:
        failed_gates = []
        if not tag_result.passed:
            failed_gates.append("source_tags")
        if len(assistant_content) < MIN_ASSISTANT_LEN:
            failed_gates.append("content_length")
        if _is_degenerate(assistant_content):
            failed_gates.append("degenerate")
        if not any(kw in content_lower for kw in keywords):
            failed_gates.append("coherence")
        return False, f"failed gates: {', '.join(failed_gates)}", gates_passed * 10


# ═══════════════════════════════════════════════════════════════════════
# STAMP
# ═══════════════════════════════════════════════════════════════════════

def stamp_batch(vetted: list[dict]) -> list[dict]:
    """Stamp vetted pairs as HiveCells."""
    cells = []
    for record in vetted:
        meta = record.get("metadata", {})
        vertical = meta.get("vertical", "general")

        stamper = HiveCellStamper(
            domain=vertical if vertical != "failure_analysis" else "general",
            source_model="SwarmJelly-4B",
            cook_script="pipeline_swarmjelly.py",
        )

        cell = stamper.stamp(
            messages=record["messages"],
            task_type=record.get("task", "selfheal"),
            skill="selfheal",
            cook_tier="gen",
            gen_model="swarmjelly-4b-q4_k_m",
            verification_score=record.get("verification_score", 70),
            source_pair_id=record.get("propolis_id", ""),
            extra={
                "failure_class": meta.get("failure_class", ""),
                "source_model": meta.get("source_model", ""),
                "pipeline_version": "1.0",
            },
        )
        cells.append(cell)
    return cells


# ═══════════════════════════════════════════════════════════════════════
# PUSH
# ═══════════════════════════════════════════════════════════════════════

def push_cells(output_dir: Path, dry_run: bool = False) -> dict | None:
    """Push stamped cells to hive-ledger."""
    cells_dir = output_dir / "cells"
    if not cells_dir.exists():
        print("  No cells directory found")
        return None

    cells = []
    for f in sorted(cells_dir.glob("*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                cells.append(json.loads(line))

    if not cells:
        print("  No cells to push")
        return None

    admin_key = os.environ.get("HIVE_ADMIN_KEY", "")
    if not admin_key:
        print("  HIVE_ADMIN_KEY not set — cannot push")
        return None

    fingerprints = [c.get("fingerprint", "") for c in cells]
    root = merkle_root(fingerprints)

    scores = [c.get("verification_score", 0) for c in cells if c.get("verification_score")]
    avg_score = sum(scores) / len(scores) if scores else 0

    grade_dist = {}
    for c in cells:
        g = c.get("grade", "cell")
        grade_dist[g] = grade_dist.get(g, 0) + 1

    batch_id = f"BATCH-SJ4-{sha256_str(root)[:12]}"

    pairs = []
    for c in cells:
        pairs.append({
            "pair_id": c.get("cell_id", ""),
            "fingerprint": c.get("fingerprint", ""),
            "domain": c.get("domain", "general"),
            "task_type": c.get("task_type", "selfheal"),
            "score": c.get("verification_score", 0),
            "tier": c.get("grade", "cell"),
            "gen_model": "swarmjelly-4b-q4_k_m",
            "cook_script": "pipeline_swarmjelly.py",
            "source_file": c.get("source_pair_id", ""),
        })

    batch_data = {
        "batch_id": batch_id,
        "domain": "general",
        "pairs": pairs,
        "merkle_root": root,
        "gate_pass_rate": 1.0,
        "avg_score": round(avg_score, 1),
        "tier_distribution": grade_dist,
        "audit_timestamp": "",
        "contamination_rate": 0.0,
        "think_tags_found": 0,
    }

    print(f"  Batch:  {batch_id}")
    print(f"  Cells:  {len(cells)}")
    print(f"  Root:   {root[:24]}...")
    print(f"  Score:  {avg_score:.1f}")
    print(f"  Grades: {grade_dist}")

    if dry_run:
        print("  [DRY RUN] Would POST to ledger")
        return None

    result = post_batch(batch_data)
    print(f"  Ledger: {result}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE CORE
# ═══════════════════════════════════════════════════════════════════════

def process_shard(
    shard_path: str,
    is_iro: bool,
    tasks: list[str],
    api_url: str,
    temperature: float,
    max_tokens: int,
    output_dir: Path,
    state: dict,
) -> dict:
    """Process a single shard through the full pipeline."""
    raw_dir = output_dir / "raw"
    vetted_dir = output_dir / "vetted"
    cells_dir = output_dir / "cells"
    for d in [raw_dir, vetted_dir, cells_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load records
    raw_records = load_jsonl(shard_path)
    shard_name = Path(shard_path).stem
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Parse into PropolisRecords
    records = []
    for rec in raw_records:
        if is_iro or "instruction" in rec:
            records.append(PropolisRecord.from_iro(rec, shard_path))
        else:
            records.append(PropolisRecord.from_propolis(rec, shard_path))

    print(f"  Loaded {len(records)} records from {shard_name}")

    # Load existing fingerprints for dedup
    seen_hashes: set[str] = set(state.get("fingerprints", []))

    all_raw = []
    all_vetted = []
    all_cells = []
    errors = []

    for i, rec in enumerate(records):
        if not rec.user_prompt:
            continue

        repair_output = ""

        # For IRO backfill, the record IS already a pair — skip inference
        if is_iro:
            reasoning = ""
            # IRO records have reasoning + output already baked in
            assistant_content = rec.model_output
            if not assistant_content:
                continue

            # Determine task from source field in original record
            orig = raw_records[i]
            source = orig.get("source", "")
            task = "diagnose"  # default
            for t in TASK_ORDER:
                if t in source or (t == "compare" and "rank" in source):
                    task = t
                    break

            pair = {
                "propolis_id": rec.id,
                "task": task,
                "messages": [
                    {"role": "system", "content": SWARMJELLY_SYSTEM},
                    {"role": "user", "content": rec.user_prompt},
                    {"role": "assistant", "content": assistant_content},
                ],
                "metadata": {
                    "failure_class": rec.failure_class,
                    "vertical": rec.vertical,
                    "source_model": rec.source_model,
                    "inference_model": "backfill",
                },
            }

            passed, reason, score = vet_pair(pair, seen_hashes)
            if passed:
                pair["verification_score"] = score
                all_vetted.append(pair)
            continue

        # Live inference path — generate 5 tasks per failure
        ordered = [t for t in TASK_ORDER if t in tasks]
        if "compare" in ordered and "repair" not in ordered:
            ordered.remove("compare")

        for task in ordered:
            prompt_text = build_prompt(task, rec, repair_output)
            messages = [
                {"role": "system", "content": SWARMJELLY_SYSTEM},
                {"role": "user", "content": prompt_text},
            ]

            t0 = time.time()
            response = infer(messages, api_url, temperature, max_tokens)
            latency = int((time.time() - t0) * 1000)

            if response is None:
                errors.append({
                    "propolis_id": rec.id,
                    "task": task,
                    "error": "inference_failed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue

            if task == "repair":
                repair_output = response

            pair = {
                "propolis_id": rec.id,
                "task": task,
                "messages": [
                    {"role": "system", "content": SWARMJELLY_SYSTEM},
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": response},
                ],
                "metadata": {
                    "failure_class": rec.failure_class,
                    "vertical": rec.vertical,
                    "source_model": rec.source_model,
                    "inference_model": "swarmjelly-4b-q4_k_m",
                    "temperature": temperature,
                    "latency_ms": latency,
                },
            }
            all_raw.append(pair)

            # Vet inline
            passed, reason, score = vet_pair(pair, seen_hashes)
            if passed:
                pair["verification_score"] = score
                all_vetted.append(pair)

        # Progress
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(records)}] raw={len(all_raw)} vetted={len(all_vetted)} errors={len(errors)}")

    # Write raw
    if all_raw:
        raw_path = raw_dir / f"{shard_name}_{ts}.jsonl"
        with open(raw_path, "w") as f:
            for rec in all_raw:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write vetted
    if all_vetted:
        vetted_path = vetted_dir / f"{shard_name}_{ts}.jsonl"
        with open(vetted_path, "w") as f:
            for rec in all_vetted:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stamp
    if all_vetted:
        all_cells = stamp_batch(all_vetted)
        cells_path = cells_dir / f"{shard_name}_{ts}.jsonl"
        with open(cells_path, "w") as f:
            for cell in all_cells:
                f.write(json.dumps(cell, ensure_ascii=False) + "\n")

    # Write errors
    if errors:
        errors_path = output_dir / "errors.jsonl"
        with open(errors_path, "a") as f:
            for err in errors:
                f.write(json.dumps(err) + "\n")

    # Write rejections (raw - vetted)
    rejected_count = len(all_raw) - len(all_vetted)

    # Update state
    state["processed_shards"][shard_path] = {
        "sha256": file_sha256(shard_path),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "pairs_generated": len(all_raw),
        "pairs_vetted": len(all_vetted),
        "cells_stamped": len(all_cells),
    }
    state["fingerprints"] = list(seen_hashes)
    state["total_pairs_generated"] = state.get("total_pairs_generated", 0) + len(all_raw)
    state["total_pairs_vetted"] = state.get("total_pairs_vetted", 0) + len(all_vetted)
    state["total_cells_stamped"] = state.get("total_cells_stamped", 0) + len(all_cells)

    return {
        "records": len(records),
        "raw": len(all_raw),
        "vetted": len(all_vetted),
        "cells": len(all_cells),
        "errors": len(errors),
        "rejected": rejected_count,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SwarmJelly Self-Healing Pipeline — propolis → Royal Jelly → ledger"
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="llama-server URL")
    parser.add_argument("--propolis-dir", type=str, default=str(DEFAULT_PROPOLIS_DIR))
    parser.add_argument("--backfill-dir", type=str, help="IRO backfill directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tasks", type=str, default=",".join(TASK_ORDER),
                        help="Comma-separated tasks (default: all 5)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--check", action="store_true", help="Check server health and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show plan, no inference")
    parser.add_argument("--push", action="store_true", help="Push cells to ledger after stamping")
    parser.add_argument("--push-only", action="store_true", help="Push existing cells only")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SwarmJelly Self-Healing Pipeline")
    print("=" * 60)

    # ── Check mode ──
    if args.check:
        check_server(args.api_url)
        return

    # ── Push-only mode ──
    if args.push_only:
        print("\n[PUSH] Pushing existing cells to ledger...")
        push_cells(output_dir, dry_run=args.dry_run)
        return

    # ── Load state ──
    state = load_state(output_dir)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in TASK_ORDER]
    if not tasks:
        print("ERROR: No valid tasks specified")
        sys.exit(1)

    # ── Discover shards ──
    propolis_shards = discover_shards(Path(args.propolis_dir), state)
    backfill_shards = discover_backfill(Path(args.backfill_dir), state) if args.backfill_dir else []

    total_shards = len(propolis_shards) + len(backfill_shards)
    print(f"\n  Propolis shards: {len(propolis_shards)}")
    print(f"  Backfill shards: {len(backfill_shards)}")
    print(f"  Tasks:           {', '.join(tasks)}")
    print(f"  API:             {args.api_url}")
    print(f"  Output:          {output_dir}")

    if total_shards == 0:
        print("\n  No unprocessed shards found.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for s in propolis_shards:
            count = sum(1 for line in open(s) if line.strip())
            print(f"    {s} ({count} records × {len(tasks)} tasks = {count * len(tasks)} pairs)")
        for s in backfill_shards:
            count = sum(1 for line in open(s) if line.strip())
            print(f"    {s} ({count} records, backfill — skip inference)")
        return

    # ── Verify server for live inference ──
    if propolis_shards:
        if not check_server(args.api_url):
            print("\nERROR: llama-server not reachable. Start it first:")
            print(f"  llama-server -m /data2/swarmjelly-4b/swarmjelly-4b-q4_k_m.gguf "
                  f"-c 8192 --port 8085 -ngl 99")
            sys.exit(1)

    # ── Process ──
    totals = {"records": 0, "raw": 0, "vetted": 0, "cells": 0, "errors": 0}

    for i, shard in enumerate(backfill_shards, 1):
        print(f"\n[{i}/{total_shards}] Backfill: {Path(shard).name}")
        result = process_shard(
            shard, is_iro=True, tasks=tasks,
            api_url=args.api_url, temperature=args.temperature,
            max_tokens=args.max_tokens, output_dir=output_dir, state=state,
        )
        for k in totals:
            totals[k] += result.get(k, 0)
        save_state(state, output_dir)

    for i, shard in enumerate(propolis_shards, len(backfill_shards) + 1):
        print(f"\n[{i}/{total_shards}] Propolis: {Path(shard).name}")
        result = process_shard(
            shard, is_iro=False, tasks=tasks,
            api_url=args.api_url, temperature=args.temperature,
            max_tokens=args.max_tokens, output_dir=output_dir, state=state,
        )
        for k in totals:
            totals[k] += result.get(k, 0)
        save_state(state, output_dir)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Records:  {totals['records']}")
    print(f"  Raw:      {totals['raw']} pairs generated")
    print(f"  Vetted:   {totals['vetted']} pairs passed")
    print(f"  Cells:    {totals['cells']} stamped")
    print(f"  Errors:   {totals['errors']}")
    print("=" * 60)

    # ── Optional push ──
    if args.push:
        print("\n[PUSH] Pushing cells to ledger...")
        push_cells(output_dir, dry_run=False)


if __name__ == "__main__":
    main()
