# Source Code Audit — RecallFace Baseline

Audit scope: public `main` source tree, especially Phase 1, 4–6, 8, 9–11, configuration, requirements and Git ignore rules. This is a static source inspection; no weights were loaded and no inference was executed.

## What is present

- Phase 1 DiffAE loader and inference wrapper.
- Phase 2 starter-set construction utilities.
- Phase 3 selection aggregator.
- Phase 4 512-D DirectionProjector.
- Phase 5 512-D ExplorationSampler.
- Phase 6 simulated-witness session code and DiffAE candidate generator.
- Phase 7 navigation training/evaluation code.
- Phase 8 evaluation scripts and a metrics JSON.
- Phase 9 Streamlit application.
- Phase 10 SQLite session/database layer.
- Phase 11 integration model loader, navigation engine and integration test.
- `config.py` and a minimal unpinned `requirements.txt`.

## Important findings and risks

### 1. Model/data assets are intentionally excluded from Git

The root `.gitignore` excludes `pretrained/diffae/`, `checkpoints/`, `*.pt`, `*.ckpt`, `*.pkl`, generated images, starter-set outputs, raw/processed data and SQLite files.

As a result, a fresh clone cannot be assumed to have:
- the official DiffAE source checkout;
- `pretrained/diffae/checkpoints/last.ckpt`;
- `pretrained/diffae/checkpoints/latent.pkl`;
- `data/starter_set/starter_xT.pt`;
- Phase 7 projector/sampler checkpoints;
- generated starter-set images.

These assets need a local inventory and documented acquisition/setup procedure. Never commit large weights or private data just to make the Git tree appear complete.

### 2. DiffAE source path is platform-specific

`phase1_diffae/load_model.py` searches only:
- `/content/diffae` (Colab);
- `/kaggle/working/diffae` (Kaggle).

It does not search the repository-relative `pretrained/diffae` path that is declared in the candidate-generator file, nor a Windows path. This might be fine in the original Kaggle environment, but is a portability risk. Do not change the baseline path during this audit; the new research workspace should document its own path configuration.

### 3. Phase 11 checkpoint discovery runs at import time

`phase11_integration/model_loader.py` calls `find_checkpoint()` at module import while defining `PROJECTOR_CHECKPOINT` and `SAMPLER_CHECKPOINT`. Missing checkpoints therefore prevent importing the module, even when a caller only wants to access other loading logic. This couples importability to local checkpoint presence.

For new RecallFace 2.0 research code, load/validate optional assets explicitly when the relevant model is requested, rather than in module-level statements.

### 4. Existing integration test verifies wiring/shapes, not realistic generation

`phase11_integration/test_integration.py` uses `torch.randn(512)` as the current latent. The test checks that twelve `(256, 256, 3)` images and finite candidate latents are produced. This is a useful integration smoke test, but it does not establish that the initial latent represents the learned DiffAE distribution, that faces are realistic, or that navigation improves witness matching.

Keep this test as an integration check; add a separate quality benchmark using fixed, valid starter/distribution latents.

### 5. Candidate generation uses one fixed stochastic tensor

`DiffAECandidateGenerator.generate()` expands the same saved `x_T` across all candidates. This improves reproducibility and makes controlled comparisons easier, but it means candidate diversity in a round primarily comes from differences in the 512-D conditioning latent. Preserve this behavior in the baseline; any different-noise experiment should be a separate ablation.

### 6. New diffusion models do not accept the same latent contract

The current DiffAE contract is a 512-D conditioning vector decoded with DiffAE's own model and noise tensor. SDXL-oriented methods and identity adapters use different conditioning and latent spaces. Their weights cannot be dropped into `DiffAECandidateGenerator.generate(candidate_latents)` as a direct replacement.

A bridge must be tested explicitly. Practical possibilities include:
- image-to-image refinement using the current selected face as image input;
- image/reference identity conditioning using the current selected candidate as an anchor;
- converting Projector/Sampler navigation output into learned or estimated attribute/structure controls;
- later, training a small Navigation-to-Condition Adapter on generated/controlled pairs.

The correct first prototype should be selected by experiments; do not claim that the 512-D RecallFace latent directly maps to SDXL semantics without evidence.

### 7. Identity-conditioned generators have a task-fit caveat

Methods that accept a reference face for ID conditioning may help preserve the *currently selected candidate* through editing. They do not automatically solve initial witness-driven identity search, and an ID embedding of a synthetic candidate can preserve that candidate's errors as well as its useful structure. Assess edit fidelity and identity drift together.

### 8. Requirements are not reproducibly pinned

`requirements.txt` lists top-level packages without versions. The Phase 1 loader explicitly imports `pytorch_lightning`, but it is not listed in the repository's requirements file. Official DiffAE dependencies may also need to be installed separately. Do not blindly pin packages before inspecting the environment in which Phase 11 last passed; first capture the working versions, Python version, Torch/CUDA versions and the DiffAE checkout revision.

### 9. Phase 8 metrics need careful interpretation

The tracked JSON includes a reference to an absolute Windows path from a local machine and reports retrieval/similarity-style round metrics. These are useful context for navigation evaluation, but they are not measurements of hair realism, smile quality, visual artifacts, or diffusion-generation performance.

## Recommended actions

1. Preserve `main` and do experiments on `recallface-2.0-research-bootstrap`.
2. Run the read-only environment diagnostic in this branch.
3. Inventory what weights/starter assets exist on the user's Kaggle/Colab/local storage.
4. Reproduce the current baseline with a valid fixed starter latent if assets are available.
5. Build diffusion-only and controlled-edit benchmarks separately from the baseline.
6. Choose a generator/bridge only after comparing quality, identity, control, latency and peak VRAM on matched cases.

## Audit status

- Static code interfaces reviewed: yes.
- Existing tests executed in this environment: no.
- Model weights loaded: no.
- Generation quality experimentally measured: no.
- Full application integration approved: no.
