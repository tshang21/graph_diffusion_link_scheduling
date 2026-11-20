"""Train the UGNN denoiser on a samples file.

    python -m ugnn.train --data <samples.pt> --name <run> --results-dir <dir> [--epochs 300 ...]

Splits the file's ENTRIES (graph, delta pairs -- NOT distinct graphs, so a graph rolled at several
deltas can land on both sides and val loss reads optimistic; judge generalisation on a held-out
graph set instead) -> AdamW on the DDPM loss, batches of equal-K blocks ->
<results-dir>/<timestamp>_<name>/<exp>.pt.  With W&B (default) every epoch is logged and the final
model is uploaded as artifact <exp>; the local copy is deleted unless --keep-local-model
(get it back with ``python -m ugnn.model --name <run>``).
"""
import argparse
import os
import random
from datetime import datetime

import torch
import wandb

from ugnn.dataset import has_delta, iter_batches, load_samples
from ugnn.model import PROJECT, build_diffusion, build_ugnn, fetch_model


def run_epoch(diffusion, batches, device, optimizer=None, accum=1):
    """Mean loss over the batches; steps the optimizer every ``accum`` batches when one is given.

    accum>1 sums the gradient over several batches before stepping, so a model too big to hold one
    full batch on the GPU still trains on the same effective batch as a smaller model."""
    diffusion.train(optimizer is not None)
    losses = []
    if optimizer:
        optimizer.zero_grad()
    with torch.set_grad_enabled(optimizer is not None):
        for i, data in enumerate(batches, 1):
            loss = diffusion.training_loss(data.to(device))
            if optimizer:
                (loss / accum).backward()
                if i % accum == 0:
                    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="samples .pt")
    p.add_argument("--name", required=True, help="run tag; folder / model / W&B artifact = <timestamp>_<name>")
    p.add_argument("--results-dir", default="runs")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=200, help="examples per (single-graph) batch")
    p.add_argument("--accum", type=int, default=1,
                   help="batches per optimizer step (gradient accumulation). A trajectory yields "
                        "ceil((T-window+1)/batch-size) batches; set batch-size and accum so their "
                        "product covers one trajectory and the effective batch matches a smaller model.")
    p.add_argument("--window", type=int, default=1,
                   help="consecutive slots per training example (1 = i.i.d. slots). >1 trains on short "
                        "TRAJECTORIES so the model can learn the service rotation a min-rate delta implies; "
                        "pick window >= 1/delta_min so the constraint is expressible inside one window.")
    p.add_argument("--stride", type=int, default=None,
                   help="slots between window starts (default: = window, non-overlapping). 1 slides.")
    p.add_argument("--windows", type=int, default=None,
                   help="keep only this many windows per trajectory (default: all). --windows 1 on a "
                        "3-delta grid = 3 training examples per graph; the subset is seeded, so it is "
                        "the same every epoch.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--train-fraction", type=float, default=0.9, help="graph-level split; any remainder is held out")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--gnn-k", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="overwrite <exp>_last.pt every N epochs (model + optimizer + epoch), so a crash or "
                        "hang costs at most N epochs; continue with --resume <exp_dir>/<exp>_last.pt")
    p.add_argument("--save-every-n-epochs", type=int, default=0, help="also save <exp>_epoch<N>.pt (0 = final only)")
    p.add_argument("--keep-local-model", action="store_true", help="keep the .pt files after the W&B upload")
    p.add_argument("--resume", default=None,
                   help="checkpoint to continue from (path or W&B run name). --epochs is the TOTAL, so "
                        "resuming a 400-epoch run with --epochs 600 trains 200 more. Architecture comes "
                        "from the checkpoint; data/lr/batch flags are still read from THIS command.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", default=PROJECT)
    p.add_argument("--wandb-name", default=None, help="default: the experiment folder name")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    # resuming: the saved architecture wins, so the state dict always fits the rebuilt model.
    prev = (torch.load(args.resume if os.path.exists(args.resume) else fetch_model(args.resume),
                       map_location="cpu", weights_only=False) if args.resume else None)
    if prev:
        for k in ("base_channels", "gnn_layers", "gnn_k", "window"):
            setattr(args, k, prev["args"][k])
    # checkpoints predating --resume have no "epoch"; a completed N-epoch run ended at N-1.
    start_epoch = prev.get("epoch", prev["args"]["epochs"] - 1) + 1 if prev else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = f"{datetime.now():%Y%m%d-%H%M%S}_{args.name}"
    exp_dir = os.path.join(args.results_dir, exp)
    os.makedirs(exp_dir)

    graphs = load_samples(args.data)
    order = list(range(len(graphs)))
    random.Random(args.seed).shuffle(order)
    n_train, n_val = round(len(graphs) * args.train_fraction), round(len(graphs) * args.val_fraction)
    train, val = [graphs[i] for i in order[:n_train]], [graphs[i] for i in order[n_train:n_train + n_val]]

    # samples with a per-graph delta -> condition the denoiser on it (one channel, see delta_cond);
    # recorded in args so load_trained rebuilds the same architecture.
    args.cond_channels = 1 if has_delta(graphs) else None
    model = build_ugnn(args.base_channels, args.gnn_layers, args.gnn_k, args.cond_channels, args.window)
    diffusion = build_diffusion(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if prev:
        model.load_state_dict(prev["model"])
        # checkpoints written before --resume existed carry no optimizer moments; at a constant lr
        # AdamW rebuilds them within a few steps, so continuing without them is fine.
        if "optimizer" in prev:
            optimizer.load_state_dict(prev["optimizer"])
        print(f"resumed {args.resume} at epoch {start_epoch}"
              + ("" if "optimizer" in prev else " (no optimizer state in checkpoint)"))
    deltas = sorted({float(g["delta"]) for g in graphs}) if args.cond_channels else []
    print(f"{exp}: {device} | {len(train)} train / {len(val)} val graphs | "
          f"{sum(q.numel() for q in model.parameters()):,} params | "
          + (f"delta-conditioned on {len(deltas)} values in [{deltas[0]:.4f}, {deltas[-1]:.4f}]"
             if deltas else "graph-only (no conditioning)")
          + f" | window={args.window} slot{'s' if args.window > 1 else ''}/example")
    run = wandb.init(project=args.wandb_project, name=args.wandb_name or exp, config=vars(args)) if args.wandb else None

    def save(tag=""):
        path = os.path.join(exp_dir, f"{exp}{tag}.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "args": vars(args), "epoch": epoch, "experiment": exp}, path + ".tmp")
        os.replace(path + ".tmp", path)          # atomic: a crash mid-write never leaves a corrupt checkpoint
        print(f"saved {path}")
        return path

    for epoch in range(start_epoch, args.epochs):
        metrics = {"train_loss": run_epoch(diffusion, iter_batches(train, args.batch_size, shuffle=True,
                                                                  window=args.window, stride=args.stride, windows=args.windows,
                                                                  seed=args.seed), device, optimizer, args.accum)}
        if val:
            metrics["val_loss"] = run_epoch(diffusion, iter_batches(val, args.batch_size, shuffle=False,
                                                                    window=args.window, stride=args.stride, windows=args.windows,
                                                                    seed=args.seed), device)
        print(f"epoch {epoch}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if run:
            run.log(metrics, step=epoch)
        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0 and epoch + 1 < args.epochs:
            save("_last")
        if args.save_every_n_epochs and (epoch + 1) % args.save_every_n_epochs == 0 and epoch + 1 < args.epochs:
            save(f"_epoch{epoch + 1}")
    final = save()
    if run:
        artifact = wandb.Artifact(exp, type="model")
        artifact.add_file(final)
        run.log_artifact(artifact)
        run.finish()                                   # blocks until the artifact upload completes
        if not args.keep_local_model:
            for f in os.listdir(exp_dir):
                os.remove(os.path.join(exp_dir, f))
            print(f"model lives on W&B as artifact {exp}; local copies removed")


if __name__ == "__main__":
    main()
