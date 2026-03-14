# SwarmJelly-4B — Build Specification

## Identity

**SwarmJelly-4B** is the self-healing intelligence model for the SwarmHive. It learns from agent failures to generate Royal Jelly — self-healing training pairs that teach models what NOT to say and how to fix mistakes.

## Mission

Most AI datasets train models what to say. Royal Jelly trains models:
- What NOT to say
- How to fix mistakes
- How to detect failure modes
- How to prevent recurring failures

## Failure Taxonomy (7 Dominant LLM Failure Modes)

```
FAILURE
│
├── reasoning
│   ├── missing_step        — jumps from premise to conclusion
│   └── false_assumption    — answers based on incorrect premises
│
├── knowledge
│   ├── hallucination       — invents facts, sources, or mechanisms
│   └── overgeneralization  — applies rules too broadly
│
├── instruction
│   ├── drift               — ignores part of the prompt
│   └── schema_break        — violates expected output format
│
└── agent
    └── tool_misuse         — fails to route to available tools
```

~80% of LLM mistakes fall into these 7 categories.

## Royal Jelly Pipeline

Each failure becomes 5 Royal Jelly training pairs:

```
failure trace
↓
SwarmJelly analysis
↓
DIAGNOSE  — What went wrong?
REPAIR    — How to fix it?
PREVENT   — What guardrail prevents this?
DETECT    — Is this output exhibiting failure mode X?
COMPARE   — Which output is better and why?
```

One failure → 5 training examples.

## Training Data Sources

| Source | Raw Count | Description |
|--------|-----------|-------------|
| selfheal | 45,417 | Royal Jelly — repair strategies, recovery patterns |
| failure | 20,829 | Failure intelligence — diagnosis, root cause |
| behavior | 38,936 | Agent workflows — behavioral patterns, execution |
| judge_traces | 164,394 | Quality evaluation trajectories |
| **Total raw** | **269,576** | |

### Dataset Balance Requirements

**CRITICAL**: No single failure type should exceed 25% of the dataset. Target distribution:

| Failure Type | Target % |
|---|---|
| hallucination | 15-20% |
| schema_break | 15-20% |
| missing_reasoning | 15-20% |
| false_assumption | 10-15% |
| instruction_drift | 10-15% |
| tool_misuse | 10-15% |
| overgeneralization | 5-10% |

Judge traces must be stratified-sampled to prevent dominance (cap at ~40% of total).

## Gold Standard Build Config (4B Tier)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Qwen/Qwen3.5-4B | Fits Jetson Orin (8GB), fast inference |
| LoRA rank | 32 | 4B tier standard |
| LoRA alpha | 16 | alpha = r/2 |
| Learning rate | 2e-5 | Higher LR for smaller model |
| Epoch fraction | 0.8 | More exposure for 4B |
| Batch | 4 × 8 = 32 effective | Standard |
| Scheduler | cosine | Standard |
| Warmup | 5% | Standard |
| Packing | True | 6x speedup |
| Precision | bf16 | Standard |
| Early stopping | patience=3, threshold=0.001 | Standard |
| AutoTokenizer | bypass Unsloth (Qwen3.5 VL dispatch bug) | Required |

## Hardware

| Stage | Hardware | Notes |
|-------|----------|-------|
| Training | RTX PRO 6000 Blackwell (96GB) GPU 1 | 4B uses ~8GB, room for large batches |
| Inference | Jetson Orin Nano 8GB (Q4_K_M GGUF) | Edge deployment |
| Inference | swarmrails CPU (bf16/Q8_0) | Backup |
| Inference | Any 8GB+ GPU (bf16) | Universal |

## Deployment Architecture

```
Cook Pipeline (vLLM 9B/27B on GPU)
  ↓ produces pairs + failures
  ↓
PropolisCollector (code, deterministic)
  ↓ captures failures to JSONL
  ↓
SwarmJelly-4B (Jetson edge OR CPU)
  ↓ failure → 5× Royal Jelly pairs
  ↓ classify + label + score
  ↓
Vet Pipeline (code, 6 gates + contamination)
  ↓ clean pairs only
  ↓
Honey Ledger (SQLite, code)
  ↓ registered + Merkle sealed
  ↓
Dashboard (FastAPI, code)
```

## Validation Gates (Pre-Cook)

Before training begins, the dataset MUST pass:

1. **Format gate** — every record has valid `messages` structure
2. **Role gate** — system→user→assistant pattern
3. **Content length** — assistant ≥50 chars, user ≥10 chars
4. **Degenerate gate** — no 40+ char repeated 3+ times
5. **Sequence length** — flag >32K char sequences
6. **Dedup gate** — zero duplicate assistant responses
7. **Source balance** — no source >50% of dataset
8. **Failure type balance** — no failure type >25% of dataset
9. **Sample inspection** — manual review of random samples

## Success Criteria

| Metric | Target |
|--------|--------|
| Training loss | < 0.5 |
| Eval loss | < 0.4 |
| Failure classification accuracy | 80%+ (held-out set) |
| Repair coherence | Manual review of 50 samples |
| Edge inference | < 100ms on Jetson Q4_K_M |
| End-to-end | propolis → SwarmJelly → vet → ledger (no contamination) |

## Files

| File | Purpose |
|------|---------|
| `cook_swarmjelly.py` | Dataset assembly — IRO→messages conversion, dedup, split |
| `validate_swarmjelly.py` | Full dataset audit — 9 validation gates |
| `train_swarmjelly_4b.py` | Training script — Gold Standard config |
| `BUILD_SPEC.md` | This file |
