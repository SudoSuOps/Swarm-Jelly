# SwarmJelly-4B — Research Log

## Build: SwarmJelly-4B v1 (2026-03-14)

**System**: SwarmJelly — Self-Healing Intelligence Model
**Base Model**: Qwen3.5-4B
**Training Method**: bf16 LoRA r=32 alpha=16 (Unsloth/SFTTrainer)
**Dataset**: 225,000 curated pairs (202,500 train / 22,500 eval)
**Hardware**: RTX PRO 6000 Blackwell 96GB @ 300W
**Steps**: 5,062 (0.8 epoch, effective batch 32)

**Core Architecture**:
```
Cook Models (9B / 27B)
    ↓
Propolis Collector
    ↓
SwarmJelly (4B)
    ↓
Royal Jelly Generator (DIAGNOSE, REPAIR, PREVENT, DETECT, COMPARE)
    ↓
Vet Pipeline (6 deterministic gates)
    ↓
Honey Ledger (SQLite + Merkle + Hedera)
    ↓
Training Loop
```

**Guiding Principle**: *"Validate the validator. Then validate the validator again."*

---

## ENTRY 001 — Validator Catastrophic Backtracking

```
Date:           2026-03-14
Component:      validate_swarmjelly.py — Degenerate Gate
Failure Type:   Validator Failure — Catastrophic Backtracking

Observation:
  The degenerate content detector used regex pattern (.{40,})\1{2,}
  to identify repeated substring patterns in training data. When run
  against 220K+ records containing long-form judge traces (some
  exceeding 10K chars), the regex engine entered catastrophic
  backtracking. A single record consumed 100% CPU for 40+ minutes
  with no termination. The validator itself became the failure.

Root Cause Hypothesis:
  Backreference patterns with unbounded quantifiers (.{40,}) create
  exponential state space in NFA-based regex engines. The Python re
  module does not implement backtracking guards. Long judge trace
  content (structured JSON evaluations with repeated field names)
  created worst-case matching conditions — near-matches that force
  the engine to explore exponentially many alignment positions before
  rejecting.

Repair Strategy:
  Three iterations attempted:
  1. Safe substring repeat check (sliding window + str.count) — still
     slow on 220K records at scale.
  2. Windowed sampling (200-char windows, skip 5000+ chars) — faster
     but added complexity for marginal value.
  3. Final: removed degenerate gate entirely. Rationale: all source
     data had already passed the upstream vet.py 6-gate pipeline,
     which includes its own degenerate detection with bounded checks.
     Re-validating with a weaker regex was redundant defense adding
     risk.

Validator Analysis:
  CRITICAL FINDING: The validator was more dangerous than the data it
  was validating. A degenerate regex in a quality gate created a
  denial-of-service condition against the build pipeline itself. The
  validator did not fail gracefully — no timeout, no progress
  indicator, no fallback. This is a class of failure where the safety
  system becomes the hazard.

Frontier Model Insight:
  Regex-based content filters are a common pattern in LLM output
  validation (format checkers, safety filters, degenerate detectors).
  Production systems using backreference patterns against unbounded
  model outputs are vulnerable to the same catastrophic backtracking.
  This is especially dangerous in streaming inference pipelines where
  a validator stall blocks the entire output path.

SwarmJelly Implication:
  SwarmJelly should learn to classify "validator self-failure" as a
  distinct failure mode. The current 7-mode taxonomy covers model
  output failures but not pipeline infrastructure failures. A
  validator that hangs, crashes, or produces false results is a
  failure the system must detect and repair.

Dataset Impact:
  YES — new Royal Jelly pair type: DETECT validator self-failure.
  Training examples should include cases where a quality gate itself
  exhibits pathological behavior (infinite loops, false positives on
  clean data, resource exhaustion). The repair strategy is
  architectural: bounded execution, timeouts, graceful degradation.

Architecture Lesson:
  Defense-in-depth does not mean re-running equivalent checks at every
  layer. Each validation layer must add unique value. If upstream
  gates already cover degenerate detection, downstream re-checking
  with a weaker implementation adds risk without adding safety.
  Validate the validator — but also validate whether the validator
  is necessary.
```

---

## ENTRY 002 — Dataset Dominance and Failure Type Scarcity

```
Date:           2026-03-14
Component:      cook_swarmjelly.py — Dataset Assembly
Failure Type:   Dataset Dominance — Source Imbalance

Observation:
  Initial dataset assembly with uncapped judge traces produced 244K
  clean pairs where judge_trace constituted 65% of the training set.
  The 7 failure modes within judge traces were also severely
  imbalanced: schema_break at 62%, hallucination at <1%. Without
  stratified sampling, the model would learn to detect format
  violations while remaining nearly blind to hallucination — the
  failure mode most dangerous in production.

Root Cause Hypothesis:
  Judge trace generation was biased toward easily-detectable failure
  modes. Schema violations are deterministic to generate (break JSON
  structure) and evaluate (parse fails = violation). Hallucination
  requires domain knowledge to construct and verify. The cook pipeline
  that produced judge traces naturally gravitates toward failure modes
  with clear ground truth, creating a systematic blind spot for
  ambiguous failure modes.

Repair Strategy:
  Two-layer stratified sampling:
  1. Source-level: --judge-cap parameter limits judge trace
     contribution to prevent source dominance.
  2. Failure-type-level: within judge traces, stratified sampling
     allocates equal budget per failure type. Small buckets
     (hallucination: 1,462, cascading_error: 386) are taken in full.
     Large buckets (missing_step: 43,062) are down-sampled.

  Final 225K dataset composition: judge 65%, behavior 16%, selfheal
  10%, failure 9%. Judge dominance persists because non-judge sources
  only contribute ~85K clean pairs. The mathematical constraint:
  225K target with 85K non-judge = minimum 62% judge. Accepted as
  inherent to available data volume.

Validator Analysis:
  The validate_swarmjelly.py source balance gate (no source >50%)
  was defined in BUILD_SPEC.md but not implemented in the validator
  code. The validator would PASS a dataset that violates the spec's
  own balance requirements. This is a specification-implementation
  drift — the spec documents a constraint the system does not
  enforce.

Frontier Model Insight:
  Dataset imbalance is a well-known problem, but the mechanism here
  is instructive: the bias originates in the ease of generating
  ground truth for different failure types. Deterministic failures
  (format violations, tool call errors) are cheap to generate and
  verify. Semantic failures (hallucination, overgeneralization) are
  expensive. Without active correction, any automated failure
  generation pipeline will over-represent the cheapest-to-detect
  failure modes.

SwarmJelly Implication:
  SwarmJelly's Royal Jelly multiplier (5x tasks per failure) should
  weight scarce failure types higher. When the system encounters a
  hallucination failure, it should generate more than 5 training
  pairs — perhaps 10-15 — to compensate for the natural scarcity
  of hallucination examples in the pipeline.

Dataset Impact:
  YES — priority pair generation for underrepresented failure modes.
  Hallucination (0.9% of judge traces) and cascading_error (0.2%)
  need dedicated cook runs to increase representation. The stratified
  sampler masks the problem but doesn't solve it.

Architecture Lesson:
  Stratified sampling is a treatment, not a cure. It rebalances
  existing data but cannot create signal that was never captured. The
  real fix is upstream: the failure generation pipeline must be tuned
  to produce hard-to-detect failure modes, not just easy ones. Build
  the pipeline to seek what's rare, not amplify what's common.
```

---

## ENTRY 003 — Architecture Misclassification Disabling Packing

```
Date:           2026-03-14
Component:      Qwen3.5-4B — Unsloth Training Pipeline
Failure Type:   Architecture Misclassification — Packing Disabled

Observation:
  Unsloth 2026.3.4 detected Qwen3.5-4B as a vision-language model
  and disabled sample packing: "Sample packing skipped
  (vision-language model detected)." This is the same VL dispatch
  bug documented across all Qwen3.5 builds. The model architecture
  (Qwen3_5ForConditionalGeneration) inherits from the VL class even
  for text-only variants. Without packing, training time increases
  approximately 6x (estimated 29 hours vs ~5 hours).

Root Cause Hypothesis:
  Qwen3.5 uses a unified architecture class
  (Qwen3_5ForConditionalGeneration) for both text and vision-language
  variants. The config.json includes vision_config even in text-only
  models (required for weight loading — removing it causes shape
  assertion errors in qwen3_vl.py). Unsloth's model type detection
  reads the architecture string and config keys, correctly identifying
  the VL class but incorrectly inferring VL behavior. The model is
  architecturally VL but functionally text-only.

Repair Strategy:
  For this build: accept the 6x slowdown. The 96GB RTX PRO 6000 has
  capacity headroom — 31GB used of 96GB available. Training completes
  in ~29 hours, which is acceptable for a one-time build.

  For future builds: investigate Unsloth source to find the packing
  gate and patch the VL detection to check for actual vision inputs
  in the dataset, not just architecture class. Alternative: use
  AutoTokenizer bypass already in script to force text-mode packing.

Validator Analysis:
  No validator caught this. The training script logs the config
  correctly (packing=True in config) but Unsloth silently overrides
  it. The "Sample packing skipped" message appears in logs but is
  not treated as an error or warning condition. A pre-training
  validator should verify that critical config parameters (packing,
  precision, gradient checkpointing) are actually applied, not just
  requested.

Frontier Model Insight:
  Unified architecture classes that span modalities (text + vision +
  audio) are becoming standard (GPT-4o, Gemini, Qwen3.5). Training
  tooling built on architecture-string detection will increasingly
  misclassify models. The relevant signal is not what a model CAN do
  architecturally but what it IS DOING in this specific training run.

SwarmJelly Implication:
  Config-vs-runtime divergence is a failure mode SwarmJelly should
  detect. When a training system requests packing=True but the
  framework silently disables it, the training is correct but
  suboptimal. This class of "silent degradation" — where nothing
  fails but performance is materially worse — is difficult to detect
  without explicit runtime assertions.

Dataset Impact:
  YES — new Royal Jelly pair type: configuration drift detection.
  Training examples where a system's actual behavior diverges from
  its declared configuration, and the divergence has measurable
  impact on outcomes.

Architecture Lesson:
  Silent overrides are more dangerous than loud failures. A system
  that crashes on misconfiguration forces immediate diagnosis. A
  system that silently degrades allows suboptimal behavior to persist
  undetected. Every configuration override should be logged at
  WARNING level with the performance impact quantified.
```

---

## ENTRY 004 — Scale Validation of Parallel Registration

```
Date:           2026-03-14
Component:      Honey Ledger — Judge Trace Registration
Failure Type:   Scale Validation — 24-Worker Parallel Vet

Observation:
  164,326 judge traces were vetted and registered into the Honey
  Ledger using 24 parallel workers in 2,607 seconds (43 minutes).
  Each worker loaded the full fingerprint set (1.37M existing pairs)
  into memory for deduplication. Total rejection rate: 68/164,394
  (0.04%). Rejection breakdown: 28 degenerate, 3 invalid JSON output,
  3 cross-chunk duplicates, 34 other. The 99.96% pass rate validates
  the upstream vet pipeline's effectiveness.

  Quality distribution of registered judge traces:
  - Honey tier (80+): 81,589 (64%)
  - Cluster tier (60-79): 46,249 (36%)
  - Average score: 77.2

  Memory footprint: each worker consumed ~2.7GB (1.37M fingerprints
  x hash + metadata). 24 workers x 2.7GB = ~65GB peak RAM on a
  256GB system (25% utilization).

Root Cause Hypothesis:
  N/A — this is a success observation documenting scale behavior.
  The 0.04% rejection rate after upstream vetting confirms that the
  6-gate vet pipeline is effective at the batch level. The small
  number of rejections (68) represent edge cases that passed upstream
  gates individually but were caught by cross-batch dedup or
  stricter thresholds in the registration layer.

Repair Strategy:
  None needed for current scale. For 10x scale (1.5M+ records per
  registration batch), the per-worker memory model (full fingerprint
  set per worker) will not scale linearly. Consider: bloom filter
  for first-pass dedup (O(1) memory per lookup), with full hash
  verification only for bloom-positive matches.

Validator Analysis:
  The two-layer validation architecture (vet pipeline -> registration
  gates) performed as designed. The vet pipeline handles content
  quality (format, roles, degenerate, contamination). The registration
  layer handles identity integrity (fingerprint dedup, Merkle sealing,
  score assignment). Separation of concerns is correct: content
  validators should not know about ledger state, and the ledger should
  not re-evaluate content quality.

Frontier Model Insight:
  At 1.5M pairs, the Honey Ledger represents a non-trivial data
  provenance system. The Merkle root per batch provides tamper
  evidence. The fingerprint dedup prevents training data
  contamination across builds. These properties — provenance,
  integrity, deduplication — are rarely implemented in open model
  training but are essential for claiming data quality at
  institutional grade.

SwarmJelly Implication:
  SwarmJelly-generated Royal Jelly pairs must pass the same
  registration path. The ledger does not distinguish between
  human-curated and model-generated pairs after registration — both
  are fingerprinted, scored, and sealed equally. This means
  SwarmJelly's output quality directly determines ledger quality.
  The system is self-reinforcing: better SwarmJelly -> better Royal
  Jelly -> better training data -> better next-generation SwarmJelly.

Dataset Impact:
  The 127,838 registered judge traces are now available as a training
  source for future builds. Domain embedding analysis shows:
  aviation 38%, medical 28%, CRE 14% — the judge traces carry
  cross-domain signal that enriches vertical-specific models.

Architecture Lesson:
  Parallel registration with independent workers and shared-nothing
  fingerprint sets is correct for write-heavy workloads. The
  cross-chunk duplicate count (3) confirms that inter-worker
  collision is rare enough to handle in a post-registration sweep
  rather than requiring distributed locking during registration.
  Optimize for throughput, reconcile for correctness.
```

---

## ENTRY 005 — Thermal Strategy for Long-Duration Training

```
Date:           2026-03-14
Component:      Power Management — Fleet Thermal Strategy
Failure Type:   Operational — Efficiency Optimization

Observation:
  Training power limits were manually set below TDP:
  - RTX PRO 6000 Blackwell: 300W (TDP 350W, -14%)
  - RTX 3090: 275W (TDP 370W, -26%)

  At 300W the RTX PRO 6000 maintains 100% utilization at 75C with
  ~21s/step. The 3090 at 275W maintains 89% utilization at 71C
  running inference. CPU utilization on both rigs: <5%.

  The power reduction trades single-step throughput for thermal
  stability over 29-hour training runs. The relationship between
  power and throughput is non-linear: a 14% power reduction on the
  6000 likely costs <5% throughput due to GPU boost clock behavior
  (the card throttles less frequently at sustained lower power,
  maintaining more consistent step times).

Root Cause Hypothesis:
  Long-duration training runs are thermally constrained, not compute
  constrained. A GPU at TDP for 29 hours will thermal-throttle
  periodically, creating variable step times and potentially
  triggering thermal shutdowns. A GPU at 85% TDP runs in a stable
  thermal envelope indefinitely, with more consistent step times
  and lower failure risk.

Repair Strategy:
  N/A — this is a best practice observation. Power capping should be
  standard for any training run exceeding 4 hours.

Validator Analysis:
  No automated system validates thermal or power configuration before
  training. A pre-flight check that verifies persistence mode, power
  limits, and thermal headroom would prevent mid-training failures.

Frontier Model Insight:
  At scale (127x RTX PRO 6000 Blackwell fleet on order), power
  management becomes a fleet-level concern. 127 GPUs x 50W savings
  = 6.35kW continuous reduction. Over a 29-hour training run, that
  is 184 kWh saved per build — meaningful at datacenter scale and
  relevant to sustainability reporting.

SwarmJelly Implication:
  Operational parameters (power limits, thermal management, GPU
  persistence mode) affect training reproducibility. Two identical
  training runs at different power limits may produce different
  loss curves due to thermal throttling introducing variable
  computation precision. SwarmJelly builds should log power
  configuration as part of the training manifest.

Dataset Impact:
  No direct dataset impact. However, infrastructure reliability
  pairs (operational best practices, failure prevention strategies)
  could be generated as a new Royal Jelly category for DevOps/MLOps
  agent training.

Architecture Lesson:
  Infrastructure reliability is a prerequisite for training
  reliability. A training run that fails at hour 27 of 29 due to
  thermal shutdown wastes more compute than running 10% slower from
  the start. Optimize for completion probability, not peak
  throughput.
```

---

## Ledger State at Build Time

```
Honey Ledger: 1,497,397 total pairs

Domain Distribution:
  cre                     809,489    (54.1%)
  medical                 401,182    (26.8%)
  judge                   127,838    ( 8.5%)
  failure_intelligence     61,246    ( 4.1%)
  aviation                 53,428    ( 3.6%)
  agent_workflows          38,936    ( 2.6%)
  general                   5,278    ( 0.4%)

Average Scores by Domain:
  aviation                 91.3
  cre                      87.4
  medical                  84.5
  judge                    77.2
  general                  77.1
  failure_intelligence     75.4
  agent_workflows          69.2
```

---

## SwarmJelly-4B Training Dataset

```
Total:     225,000 pairs
Train:     202,500
Eval:       22,500

Source Mix (train):
  judge_trace            131,812  (65.1%)
  behavior                32,373  (16.0%)
  selfheal                21,173  (10.5%)
  failure                 17,142  ( 8.5%)

Failure Taxonomy (7 modes):
  reasoning/missing_step
  reasoning/false_assumption
  knowledge/hallucination
  knowledge/overgeneralization
  instruction/drift
  instruction/schema_break
  agent/tool_misuse

Royal Jelly Tasks (5 per failure):
  DIAGNOSE  — What went wrong?
  REPAIR    — How to fix it?
  PREVENT   — What guardrail prevents this?
  DETECT    — Is this output exhibiting failure mode X?
  COMPARE   — Which output is better and why?

7 failure modes x 5 Royal Jelly tasks = 35 training signal types
```

---

*Research log maintained by SwarmJelly build system. Entries are structured
for potential compilation into technical white paper documentation.*

*"You get more Bees with Honey."*
