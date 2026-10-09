# RecallFace 2.0 — Controlled Benchmark Protocol

## Goal

Determine whether an alternative generation/editing layer improves visual detail while maintaining identity and compatibility with the existing learned navigation pipeline.

## Conditions

- A: existing DiffAE baseline.
- B: alternative diffusion generator without RecallFace navigation.
- C: selected diffusion approach with a fixed RecallFace navigation signal.
- D: navigation + diffusion + controlled candidate evolution (only after C works).

Do not compare results produced with materially different inputs, resolution, or undocumented settings.

## Fixed inputs and reproducibility

For each run, record:
- source commit and model/checkpoint revision;
- input latent or reference image identifier;
- random seed;
- generation resolution;
- sampler/scheduler and inference steps;
- precision/offloading settings;
- edit prompt, mask and conditioning strengths;
- wall-clock runtime and peak GPU memory;
- all failed/OOM runs, not just successful samples.

Use identical fixed inputs and seeds wherever a fair comparison is possible. Keep a separate qualitative set and avoid tuning on the final test set.

## Review dimensions

Score each candidate on a documented rubric:
1. Hair: silhouette, texture, hairline and boundary artifacts.
2. Eyes and eyebrows: symmetry, details and alignment.
3. Nose and cheeks: coherent shape.
4. Mouth/teeth/smile: realism and requested expression.
5. Skin and lighting: artifacts, texture and consistency.
6. Global geometry: plausible facial proportions.
7. Identity consistency: embedding similarity plus human review.
8. Requested edit accuracy.
9. Diversity among candidates.
10. Multi-round stability.
11. Latency and peak VRAM.

Automated face-embedding similarity is only a proxy; it must not be treated as definitive proof of forensic identity accuracy.

## Suggested result table

| Run ID | Condition | Model/checkpoint | Seed | Resolution | Identity metric | Edit accuracy | Quality notes | Latency | Peak VRAM | Pass/Fail |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|

## Decision rule

Choose the generator only after comparing the same cases and reviewing both visual evidence and quantitative metrics. Better-looking images alone are insufficient if identity drift or requested-edit failures increase.

## Safety and application scope

RecallFace is a research prototype for facial-composite exploration, not a validated automated identification system. Generated candidates must not be presented as proof of identity or used as the sole basis for accusing, identifying, or taking action against a person.
