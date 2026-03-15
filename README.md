# SwarmJelly-4B

Self-healing intelligence model for the SwarmHive.

**Converts propolis (failures) into Royal Jelly (self-healing training pairs).**

7 failure modes x 5 Royal Jelly tasks = 35 training signal types per failure.

## Files

| File | Purpose |
|---|---|
| `BUILD_SPEC.md` | Full build specification |
| `cook_swarmjelly.py` | Dataset assembly with stratified sampling |
| `train_swarmjelly_4b.py` | Gold Standard training script (4B tier) |
| `validate_swarmjelly.py` | Pre-cook dataset validator |

## Quick Start

```bash
# Dry run - see data distribution
python3 cook_swarmjelly.py --include-judge /data2/judge_backfill/ --judge-cap 50000 --dry-run

# Assemble balanced dataset
python3 cook_swarmjelly.py --include-judge /data2/judge_backfill/ --judge-cap 50000

# Validate
python3 validate_swarmjelly.py --all

# Train (smoke test)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 train_swarmjelly_4b.py --smoke-test

# Train (full)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 train_swarmjelly_4b.py
```

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

## Dataset Balance

Stratified sampling ensures no single failure type or source dominates:

| Source | Count | % |
|---|---|---|
| selfheal | 45,417 | 31% |
| judge_trace (capped) | 40,736 | 28% |
| behavior | 38,936 | 27% |
| failure | 20,829 | 14% |
| **Total (after dedup)** | **125,028** | |

## License

Proprietary - Swarm & Bee AI
