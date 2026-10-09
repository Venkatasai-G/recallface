# Phase 0 — Baseline Audit

## Confirmed source interfaces

These observations come from inspecting the current repository's source code.

### Projector
File: `phase4_projector/direction_projector.py`

- Input: tensor `(batch_size, 512)`
- Output: tensor `(batch_size, 512)`
- Purpose: predict a latent exploration direction.

### Sampler
File: `phase5_sampler/exploration_sampler.py`

- Inputs: latent `(batch_size, 512)`, round number `(batch_size, 1)`
- Output: positive per-dimension variance `(batch_size, 512)`
- Sampling includes Gaussian noise.

### DiffAE candidate generator
File: `phase6_simulated_data/diffae_candidate_generator.py`

- Input: NumPy array `(N, 512)`
- Output: list of RGB `uint8` arrays.
- Current code reuses a fixed `x_T` tensor for candidates and runs DiffAE rendering at its configured evaluation step count.
- Expected assets include `pretrained/diffae/checkpoints/last.ckpt` and `data/starter_set/starter_xT.pt`.

### Phase 11 navigation engine
File: `phase11_integration/navigation_engine.py`

- Loads Phase 7 projector/sampler checkpoints.
- Loads the Phase 6 DiffAE candidate generator.
- Reads `pretrained/diffae/checkpoints/latent.pkl` for empirical mean/std.
- Produces candidate latents with shape `(N,512)`, then calls the DiffAE generator.

### Configuration
File: `config.py`

- Latent dimension: 512
- Starter set size: 12
- Candidate count: 12–20
- Random seed: 2026

## Known asset dependency gap

The public source tree does not show these files/folders as tracked files:
- `pretrained/diffae/checkpoints/last.ckpt`
- `pretrained/diffae/checkpoints/latent.pkl`
- `data/starter_set/starter_xT.pt`
- `checkpoints/PHASE7_FINAL/projector_best.pt`
- `checkpoints/PHASE7_FINAL/sampler_best.pt`

They may exist locally or be intentionally ignored by Git. Confirm locally before trying a baseline inference run.

## Baseline experiments required

1. **Reconstruction sanity check:** use a known starter latent and the baseline generator.
2. **Zero/low navigation movement:** separate generator weaknesses from navigation effects.
3. **Navigation movement sweep:** compare candidate outputs across small, medium and larger movement scales.
4. **Round stability:** follow selections for multiple rounds and track visible drift.
5. **Per-region inspection:** hair, eyes, eyebrows, nose, mouth/teeth, skin and face/background boundaries.
6. Save fixed seeds, inputs, outputs, runtime, peak VRAM and all config values.

## Current hypotheses (not conclusions)

- DiffAE rendering capacity may limit high-frequency hair and facial detail.
- Navigation movement may push conditions away from well-supported regions.
- Reusing a fixed stochastic tensor improves comparison consistency but may reduce stochastic diversity.
- Several causes may coexist; the benchmark should distinguish them.

## Phase 0 exit criteria

- Required source files and external assets are inventoried.
- Current environment and GPU are recorded.
- Baseline inference can be reproduced OR the precise missing dependency is recorded.
- No changes are made to original model or application logic.
