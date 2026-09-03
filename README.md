# Einstein's Mom

A pre-registered experiment on scalable oversight: does alignment instilled by
shaping a model's *reasoning process* generalise further than alignment
instilled by grading its *outputs*, when the overseer cannot verify those
outputs?

**Nothing here has been run.** This repo contains a design document and a set of
predictions committed in advance. There is no experiment code, no training run,
and no results — by decision, so that the hypotheses cannot move to wherever the
data lands.

- [`spec.md`](spec.md) — the experimental design: hypotheses H1–H4, the two
  training arms, confounds, acceptance criteria, and the 24 GB compute budget.
- [`handoff.md`](handoff.md) — environment setup for whoever implements it.
- [`einsteins-mom-blog-post.md`](einsteins-mom-blog-post.md) — the
  pre-registration written up for a general reader.
- `flake.nix` / `flake.lock` — Nix dev shell for the PyTorch stack. Declared,
  but not yet entered or tested.

Scoped to one RTX 3090 (24 GB). If you want to run it, please do.

MIT licensed.
