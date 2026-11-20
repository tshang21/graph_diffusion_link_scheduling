"""Load a trained SAGNN policy and dual-roll it out at an arbitrary (delta, weight_decay).

The SAGNN policy (``GNN1``) is lambda-conditioned and *delta-agnostic in
training*: ``collect_expert.sawl`` trains it by sampling lambda and never uses
``delta`` during training — delta only enters the *test-time* dual update.  So
the policy saved at ``models/exp_..._delta_0.1_..._of_1.pt`` can be rolled out at
ANY test delta.  This module factors out ``collect_expert.py``'s test-phase loop
into a reusable, load-from-checkpoint function whose math mirrors it exactly
(continuous schedule + continuous conflicts drive the dual ascent; binary
schedule is recorded for objective/constraint accounting).

Use it to dual-roll the SAGNN expert at out-of-distribution deltas so it can be
compared head-to-head with the UGNN diffusion model (see ugnn/compare_ood.py).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from tqdm import tqdm

from sagnn.gnn import GNN1
from common.utils import remove_collisions


def load_sagnn(model_path: str, num_features: int = 256, num_layers: int = 3,
               batch_norm: bool = True, device="cpu") -> GNN1:
    """Rebuild ``GNN1`` with the saved policy's architecture and load weights.

    Defaults match the collect_expert training defaults (num_features=256,
    num_layers=3, norm=1) that produced models/exp_..._of_1.pt.  ``state_dict``
    is loaded strictly, so a shape mismatch fails loudly.
    """
    num_features_list = [1] + [num_features] * num_layers
    model = GNN1(num_features_list, batch_norm).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def sagnn_dual_rollout_graph(model, A, device, T: int = 200, lr_lambda: float = 2.0,
                             delta: float = 0.1, weight_decay: float = 0.05, T0: int = 1,
                             mask_collisions: bool = False, rng=None, return_time: bool = False):
    """One graph: SAGNN policy inside the state-augmented dual loop, for exactly ``T`` steps.

    Mirrors ``collect_expert.sawl``'s test phase:
        s_t      = mean over batch of policy(lambda_t, edge_index)   (continuous)
        s_bin    = (s_t >= 0.5)
        conflicts= clamp(1 - A @ s_t, 0)                              (continuous)
        every T0 steps:  lambda <- clamp(lambda - lr_lambda*((s*conflicts - delta) + wd*lambda), 0)

    Rolls the dual loop for ``T`` steps from cold ``lambda_0=0`` and records ALL of
    them (the first steps are the cold-start transient).  Discarding that transient is
    a DATASET concern, not a rollout concern: the caller drops the leading ``warmup``
    steps when building the UGNN dataset (see :func:`sagnn_dual_rollout`).  This keeps
    the rollout cost -- and the ``return_time`` measurement -- exactly ``T`` steps,
    comparable to the other generators.

    ``mask_collisions``: when True, the per-step schedule is passed through
    ``remove_collisions`` and the collision-free CONTINUOUS schedule
    (``s_masked = s * survivor_mask``) is what gets SAVED — but the dual update is
    UNCHANGED and still driven by the RAW continuous ``s`` (raw conflicts, raw
    lambda).  So masking only post-processes the recorded schedule; the dual
    dynamics (and hence the lambda trajectory) are bit-identical to the unmasked
    rollout.  ``rng`` (np.random.Generator) seeds the ``remove_collisions`` tie-breaks.

    Returns ``(raw [T,K], binary [T,K], lambda [T,K])`` on CPU (``lambda[t]`` is the
    dual variable that produced step t's schedule).  With ``return_time`` a 4th value
    is the seconds spent in the T-step loop (per-graph setup excluded).
    """
    A = torch.as_tensor(A, dtype=torch.float32, device=device)
    K = int(A.shape[0])
    edge_index = A.nonzero(as_tuple=False).t().contiguous().to(device)
    ones = torch.ones(1, K, device=device)
    lam = torch.zeros(K, 1, device=device)
    A_cpu = A.detach().cpu().numpy() if mask_collisions else None

    raws, bins, lams = [], [], []
    t0 = time.perf_counter()                                              # time the full T-step rollout
    for step in range(T):
        s = model(lam.detach(), edge_index).view(-1, K).mean(dim=0)        # (K,) continuous (RAW)
        conflicts = torch.clamp(ones - torch.matmul(A, s), min=0.0).view(K)  # RAW dynamics
        if mask_collisions:
            # masked SAGNN: all-on slots (cold/low-lambda collapse) carry no decision -> mask to zero
            mask = remove_collisions(s.detach().cpu().numpy(), A_cpu, threshold=0.5, rng=rng,
                                     zero_if_all_active=True)
            mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
            s_save = s * mask                                             # masked continuous (SAVED only)
            s_bin = mask                                                  # 0/1 survivor mask
        else:
            s_save = s
            s_bin = (s >= 0.5).float()
        raws.append(s_save.detach().cpu())
        bins.append(s_bin.detach().cpu())
        lams.append(lam.view(K).detach().cpu().clone())
        if (step + 1) % T0 == 0:
            # dual update uses RAW s -> lambda trajectory identical to unmasked
            lam = lam - lr_lambda * ((s * conflicts - delta).view(K, 1) + lam * weight_decay)
            lam = torch.clamp(lam, min=0.0)
    if torch.cuda.is_available() and lam.is_cuda:
        torch.cuda.synchronize()                                          # accurate wall time on GPU
    gen_seconds = time.perf_counter() - t0
    out = (torch.stack(raws), torch.stack(bins), torch.stack(lams))
    return (*out, gen_seconds) if return_time else out


def sagnn_dual_rollout(model, graphs, device, T: int = 200, lr_lambda: float = 2.0,
                       delta: float = 0.1, weight_decay: float = 0.05, T0: int = 1,
                       warmup_steps: int = 0, mask_collisions: bool = False,
                       seed: int = 0, progress: bool = True, return_time: bool = False):
    """Dual-roll the SAGNN policy over many graphs at (delta, weight_decay).

    Each graph is rolled for ``T`` steps; the leading ``warmup_steps`` (cold-start
    transient) are then DROPPED, so the saved trajectory is ``T - warmup_steps`` slots.
    To keep N post-warmup schedules, pass ``T = N + warmup_steps``.  Returns a list of
    per-graph dicts (``A``, ``raw_schedule``, ``thresholded_schedule``, ``lambda``);
    with ``return_time`` also returns the aligned list of per-graph rollout seconds
    (each over the full ``T`` steps).  ``mask_collisions`` enables per-step
    ``remove_collisions`` with a deterministic ``np.random.Generator(seed + index)``.
    """
    out, times = [], []
    desc = (f"SAGNN rollout (delta={delta:g}, wd={weight_decay:g}, warmup={warmup_steps}"
            f"{', masked' if mask_collisions else ''})")
    it = tqdm(graphs, desc=desc, unit="graph") if progress else graphs
    for i, g in enumerate(it):
        rng = np.random.default_rng(seed + i) if mask_collisions else None
        raw, binary, lam, gen_seconds = sagnn_dual_rollout_graph(
            model, g["A"], device, T=T, lr_lambda=lr_lambda, delta=delta,
            weight_decay=weight_decay, T0=T0, mask_collisions=mask_collisions, rng=rng,
            return_time=True,
        )
        if warmup_steps:                                                  # drop cold-start transient
            raw, binary, lam = raw[warmup_steps:], binary[warmup_steps:], lam[warmup_steps:]
        out.append({
            "A": torch.as_tensor(g["A"], dtype=torch.float32),
            "raw_schedule": raw, "thresholded_schedule": binary, "lambda": lam,
        })
        times.append(gen_seconds)
    return (out, times) if return_time else out
