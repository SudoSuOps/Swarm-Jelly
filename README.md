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
