# SwarmJelly-4B

**Self-healing intelligence model for the SwarmHive.**

Converts propolis (failures) into Royal Jelly — self-healing training pairs that teach models what NOT to say and how to fix mistakes.

Most AI datasets train models what to say. SwarmJelly trains models:
- What NOT to say
- How to fix mistakes
- How to detect failure modes
- How to prevent recurring failures

7 failure modes × 5 Royal Jelly tasks = 35 training signal types per failure.

---

## Failure Taxonomy

~80% of LLM mistakes fall into 7 dominant failure modes:

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

### Target Distribution

No single failure type should exceed 25% of the dataset:

| Failure Type | Target % |
|---|---|
| hallucination | 15–20% |
| schema_break | 15–20% |
| missing_step | 15–20% |
| false_assumption | 10–15% |
| drift | 10–15% |
| tool_misuse | 10–15% |
| overgeneralization | 5–10% |

---

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

One failure → 5 training examples. This multiplier is the core of the self-healing loop.

---

## Training Data

### Sources

| Source | Raw Count | Description |
|---|---|---|
| selfheal | 45,417 | Royal Jelly — repair strategies, recovery patterns |
| failure | 20,829 | Failure intelligence — diagnosis, root cause |
| behavior | 38,936 | Agent workflows — behavioral patterns, execution |
| judge_traces | 164,394 | Quality evaluation trajectories |
| **Total raw** | **269,576** | |

### Assembled Dataset (after dedup + stratified sampling)

| Source | Count | % |
|---|---|---|
| selfheal | 45,417 | 31% |
| judge_trace (capped) | 40,736 | 28% |
| behavior | 38,936 | 27% |
| failure | 20,829 | 14% |
| **Total** | **125,028** | |

Split: 90% train / 10% eval.

Judge traces are stratified-sampled and capped to prevent dominance (max ~40% of total).

### Data Format

All records use the `messages` chat format with the SwarmJelly system prompt:

```json
{
  "messages": [
    {"role": "system", "content": "You are SwarmJelly — the self-healing intelligence engine for the SwarmHive..."},
    {"role": "user", "content": "[failure analysis prompt]"},
    {"role": "assistant", "content": "[reasoning]\n\n[repair/diagnosis/guardrail output]"}
  ],
  "metadata": {
    "source": "selfheal",
    "vertical": "aviation",
    "score": 92.1,
    "mechanism": "missing_step",
    "fingerprint": "sha256[:32]"
  }
}
```

IRO-format records (instruction/reasoning/output) are converted to messages format during assembly. Reasoning is prepended to the assistant response to preserve chain-of-thought.

---

## Training Data Quality — Temperature Requirement

SwarmJelly trains on pairs cooked by the Swarm data factory. The quality of the training data is gated by the [JellyScore](https://github.com/SudoSuOps/virgin-jelly) pipeline. **If the upstream cook uses the wrong temperature, the training data degrades catastrophically.**

### The Evidence

Production data from aviation cook (2026-03-14, SwarmCurator-9B):

| | Temp 0.7 (wrong) | Temp 0.05 (correct) |
|---|---|---|
| **Avg JellyScore** | 77.2 | 85.9 |
| **Honey rate** | 0.7% | 93.9% |
| **reasoning_depth** | ~0.75 | ~0.996 |

Training on temp 0.7 data means training on 94.6% Pollen — commodity pairs with shallow reasoning. Training on temp 0.05 data means training on 93.9% Honey — deep trajectory-aligned reasoning.

### Why

The [cook-domain-prompts](https://github.com/SudoSuOps/cook-domian-prompts) embed 5 trajectory keywords (IDENTIFY, CALCULATE, ANALYZE, EVALUATE, RECOMMEND) and causal connectors that the JellyScore scorer checks for. At greedy temp (0.05), the model follows these literally → high score. At higher temp, the model paraphrases → scorer can't find the markers → low score.

### The Rule

Any cook script producing training data for SwarmJelly **MUST** use:

```python
"temperature": 0.05
```

This applies to `cook_swarmjelly.py` and any upstream cooker whose output feeds into the training mix. If you see Pollen-heavy batches in the training data, **check the temperature first** before blaming the model or the prompts.

See the full analysis: [virgin-jelly README](https://github.com/SudoSuOps/virgin-jelly#temperature--prompt-alignment--critical-finding)

---

## Gold Standard Training Config (4B Tier)

| Parameter | Value | Rationale |
|---|---|---|
| Base model | `Qwen/Qwen3.5-4B` | Fits Jetson Orin (8GB), fast inference |
| LoRA rank | 32 | 4B tier standard |
| LoRA alpha | 16 | alpha = r/2 |
| Target modules | q/k/v/o/gate/up/down_proj | All attention + MLP |
| Learning rate | 2e-5 | Higher LR for smaller model |
| Scheduler | cosine | Standard |
| Warmup | 5% | Standard |
| Epoch fraction | 0.8 | More exposure for 4B |
| Batch size | 4 × 8 = 32 effective | Standard |
| Max sequence length | 4,096 tokens | Standard |
| Packing | True | ~6x speedup |
| Precision | bf16 | Standard |
| Early stopping | patience=3, threshold=0.001 | Prevents overfitting |
| Eval steps | 50 | Frequent evaluation |
| Save total limit | 5 | Disk management |
| Thinking | Disabled | Compliance, not creativity |

### Tokenizer Note

`AutoTokenizer.from_pretrained()` is used instead of Unsloth's built-in tokenizer loader due to a Qwen3.5 VL dispatch bug in Unsloth that causes incorrect tokenizer class resolution.

---

## Validation Gates

Before training begins, the dataset must pass 9 gates (`validate_swarmjelly.py`):

| # | Gate | Check |
|---|---|---|
| 1 | **Format** | Every record has valid `messages` structure |
| 2 | **Role sequence** | system → user → assistant pattern |
| 3 | **Content length** | Assistant ≥ 50 chars, user ≥ 10 chars |
| 4 | **Degenerate** | No 40+ char substring repeated 3+ times |
| 5 | **Token length** | Flag sequences > 32K chars (~8K tokens) |
| 6 | **Dedup** | Zero duplicate assistant responses |
| 7 | **Source balance** | No source > 50% of dataset |
| 8 | **System prompt consistency** | Only expected system prompts present |
| 9 | **Cross-contamination** | No known bad patterns (test data leaks, etc.) |

Plus: random sample inspection for manual review.

---

## Deployment Architecture

```
Cook Pipeline (vLLM 9B/27B on GPU)
  ↓ produces pairs + failures
  ↓
PropolisCollector (deterministic)
  ↓ captures failures to JSONL
  ↓
SwarmJelly-4B (Jetson edge OR CPU)
  ↓ failure → 5× Royal Jelly pairs
  ↓ classify + label + score
  ↓
Vet Pipeline (6 gates + contamination check)
  ↓ clean pairs only
  ↓
Honey Ledger (hive-ledger)
  ↓ registered + Merkle sealed
  ↓
Dashboard
```

### Hardware

| Stage | Hardware | Notes |
|---|---|---|
| Training | RTX PRO 6000 Blackwell (96GB) — GPU 1 | 4B uses ~8GB VRAM, room for large batches |
| Inference | Jetson Orin Nano 8GB (Q4_K_M GGUF) | Edge deployment, < 100ms target |
| Inference | swarmrails CPU (bf16/Q8_0) | Backup |
| Inference | Any 8GB+ GPU (bf16) | Universal |

### Post-Training Pipeline

1. Fix vision config if needed (Qwen3.5 VL artifact)
2. Quantize: `llama-quantize /data2/swarmjelly-4b/merged swarmjelly-4b-q4_k_m.gguf Q4_K_M`
3. Deploy on Jetson edge or swarmrails CPU
4. Wire into PropolisCollector → SwarmJelly → vet pipeline → ledger

### Hive-Ledger Integration

After training completes, the script automatically pushes model lineage to hive-ledger via `POST /api/admin/model`:

```json
{
  "model_id": "SwarmJelly-4B",
  "batch_id": "BATCH-SJ4-{train_hash[:12]}",
  "training_run_id": "run-2026-03-15",
  "pairs_used": 112525,
  "loss": 0.3421,
  "trained_at": "2026-03-15T12:00:00Z"
}
```

Requires `HIVE_ADMIN_KEY` environment variable. Skips gracefully if not set.

A `MANIFEST.json` is also written to `/data2/swarmjelly-4b/` with full training metadata including failure taxonomy, data composition, hyperparameters, and hardware info.

---

## Success Criteria

| Metric | Target |
|---|---|
| Training loss | < 0.5 |
| Eval loss | < 0.4 |
| Failure classification accuracy | 80%+ (held-out set) |
| Repair coherence | Manual review of 50 samples |
| Edge inference | < 100ms on Jetson Q4_K_M |
| End-to-end | propolis → SwarmJelly → vet → ledger (no contamination) |

---

## Quick Start

```bash
# 1. Assemble dataset (dry run first)
python3 cook_swarmjelly.py --include-judge /data2/judge_backfill/ --judge-cap 50000 --dry-run
python3 cook_swarmjelly.py --include-judge /data2/judge_backfill/ --judge-cap 50000

# 2. Validate
python3 validate_swarmjelly.py --all

# 3. Train (smoke test — 500 samples, ~5 min)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 train_swarmjelly_4b.py --smoke-test

# 4. Train (pilot — 5000 samples, ~1 hr)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 train_swarmjelly_4b.py --pilot

# 5. Train (full — 125K samples, ~20 hrs on RTX 6000)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 train_swarmjelly_4b.py

# 6. Resume from checkpoint
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 train_swarmjelly_4b.py --resume /data2/swarmjelly-4b/lora/checkpoint-XXX
```

`CUDA_VISIBLE_DEVICES=1` targets GPU 1 (RTX 6000). GPU 0 (RTX 4500) runs vLLM for active cooks.

---

## Files

| File | Purpose |
|---|---|
| `BUILD_SPEC.md` | Full build specification |
| `cook_swarmjelly.py` | Dataset assembly — IRO→messages conversion, dedup, stratified sampling, 90/10 split |
| `validate_swarmjelly.py` | Pre-cook dataset audit — 9 validation gates |
| `train_swarmjelly_4b.py` | Gold Standard training script — LoRA, packing, early stopping, ledger push, manifest |

### Output Artifacts

| Path | Contents |
|---|---|
| `/data2/swarmjelly-4b/swarmjelly_train.jsonl` | Training split (~112K records) |
| `/data2/swarmjelly-4b/swarmjelly_eval.jsonl` | Eval split (~12K records) |
| `/data2/swarmjelly-4b/lora/` | LoRA adapter checkpoints |
| `/data2/swarmjelly-4b/merged/` | Merged bf16 model (base + adapter) |
| `/data2/swarmjelly-4b/logs/` | Training logs |
| `/data2/swarmjelly-4b/MANIFEST.json` | Full training manifest with metadata |

---

## Related Repos

| Repo | Role |
|---|---|
| [virgin-jelly](https://github.com/SudoSuOps/virgin-jelly) | Royal Jelly Protocol — JellyScore scoring engine, tier system, Hedera anchoring |
| [cook-domain-prompts](https://github.com/SudoSuOps/cook-domian-prompts) | RJ-aligned prompt library — 13 domains, trajectory keywords, concept terms |
| [hive-ledger](https://github.com/SudoSuOps/hive-ledger) | Provenance chain — pair registration, Merkle proofs, model lineage |
| [hive-warehouse](https://github.com/SudoSuOps/hive-warehouse) | Data marketplace — catalog, orders, delivery |
| [SwarmRadar](https://github.com/SudoSuOps/SwarmRadar) | Signal intel platform — 5-stage pipeline feeding upstream data |

---

## License

Proprietary — Swarm & Bee AI
