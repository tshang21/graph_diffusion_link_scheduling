"""UGNN: a diffusion model over link schedules, conditioned on the conflict graph and (optionally)
on a min-rate target delta.  Run everything from src/ with PYTHONPATH=.

  dataset.py   samples files {DEFAULT_SPLIT: [{A (K,K), raw_schedule (T,K), delta?}, ...], "meta"},
               windowed training batches grouped by node count K
               CLI  python -m ugnn.dataset slice|tile ...      one slot per graph / repeat slots (batching)
  model.py     build_ugnn, build_diffusion, load_trained, fetch_model (W&B model artifact -> local .pt)
               CLI  python -m ugnn.model --name <run>          prints the checkpoint path
  train.py     CLI  python -m ugnn.train --data ... --name ...  DDPM loss -> W&B model artifact
               --window/--windows/--stride pick the training blocks, --accum widens the effective
               batch when one does not fit, --resume continues a checkpoint (--epochs is the TOTAL)
  generate.py  rollout_ugnn: T i.i.d. schedules per graph (GPU batched, or the CPU pool)
               CLI  python -m ugnn.generate --ckpt ... --data ... upload a 'schedules' artifact / timing JSON
  metrics.py   aggregate_method: objective / coverage / violation / per-link rates, STRICT scoring
               (used by the notebooks)

Retired experiment code: archive/ugnn_legacy/ at the repo root (not importable).
"""
