# RecallFace 2.0 — Isolated Research Workspace

This folder is the **experimental workspace** for RecallFace 2.0. It intentionally does not modify or replace the existing phase-based RecallFace pipeline.

## Research principle

- RecallFace Projector/Sampler: navigation/search intelligence.
- Diffusion generation/editing: visual realization.
- Controlled evolution: meaningful variation of candidates.
- Evaluation gates: visual quality, identity stability, edit control, diversity, latency and VRAM.

Do not integrate a new generator into the Streamlit app until the benchmark gives evidence that it is better than the current DiffAE baseline.

## Repository layout

- `research/`: questions, architecture notes and benchmark results.
- `benchmarks/`: isolated environment and model benchmark utilities.
- `configs/`: local settings; do not commit model weights or private data.
- `reports/`: experiment tables and conclusions.

## Start

1. Read `research/PHASE_0_BASELINE_AUDIT.md`.
2. Verify required model checkpoints and starter latent files exist locally.
3. Set up a separate Python environment.
4. Run environment diagnostics:
   ```bash
   python recallface_hybrid/benchmarks/check_environment.py
   ```
5. Do not run model inference until required assets and dependencies are verified.

## Important constraints

- The source repo currently does not track model weights or generated data. The benchmark cannot run the original DiffAE without the expected external assets.
- Keep the original project as the baseline.
- Do not claim model-quality improvements before running comparative experiments.
