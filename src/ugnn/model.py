"""The graph-only UGNN denoiser and its DDPM (train) / DDIM (sample) wrapper, from the vendored
graph_signal_diffusion Hydra configs.  Unconditional: no encoder for ``data.x``; the conflict graph
enters only through ``edge_index``.  Trained models live on W&B as artifacts <timestamp>_<run name>:

    CKPT="$(python -m ugnn.model --name <run name>)"      # local path (downloaded if needed); exit 1 if absent
"""
import argparse
import os
import re
import sys
import tempfile

import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

import graph_signal_diffusion as gsd

CONF = os.path.join(os.path.dirname(gsd.__file__), "conf")
NUM_TIMESTEPS = 500
PROJECT = "ugnn_link_scheduling"


def build_ugnn(base_channels=64, gnn_layers=2, gnn_k=2, cond_channels=None, window=1):
    """cond_channels=None -> graph-only; 1 -> the denoiser also reads data.x (the delta channel).

    ``window`` is the number of consecutive slots one example carries, as signal channels."""
    cfg = OmegaConf.load(f"{CONF}/model/ugnn.yaml")
    OmegaConf.update(cfg, "config.in_channels", window)
    OmegaConf.update(cfg, "config.out_channels", window)
    OmegaConf.update(cfg, "config.base_channels", base_channels)
    OmegaConf.update(cfg, "config.gnn_config.num_layers", gnn_layers)
    OmegaConf.update(cfg, "config.gnn_config.K", gnn_k)
    OmegaConf.update(cfg, "config.embedding_config.cond.channels", cond_channels)
    OmegaConf.update(cfg, "config.embedding_config.num_timesteps", NUM_TIMESTEPS)
    return instantiate(cfg)


def build_diffusion(model, sampler="ddpm", sampling_timesteps=50):
    """eps-parameterization, L2 loss, clip_denoised (x_0 lives in [-0.5, 0.5]); DDIM is deterministic (eta 0)."""
    cfg = OmegaConf.load(f"{CONF}/diffusion/{sampler}.yaml")
    cfg.update(num_timesteps=NUM_TIMESTEPS, parameterization="eps", loss_type="l2", clip_denoised=True)
    if sampler == "ddim":
        cfg.update(sampling_timesteps=sampling_timesteps, ddim_eta=0.0)
    return instantiate(cfg, model=model)


def fetch_model(name, project=PROJECT):
    """Local path of the newest W&B model artifact <timestamp>_<name> (downloaded if needed); None if there is none."""
    api = wandb.Api()
    pattern = re.compile(rf"^\d{{8}}-\d{{6}}_{re.escape(name)}$")
    names = sorted(c.name for c in api.artifact_type("model", project=project).collections() if pattern.match(c.name))
    if not names:
        return None
    root = os.path.join(tempfile.gettempdir(), "ugnn_models", names[-1])
    return os.path.join(api.artifact(f"{project}/{names[-1]}:latest").download(root=root), f"{names[-1]}.pt")


def load_trained(ckpt, device, sampler="ddim", sampling_timesteps=50):
    """Rebuild a train.py checkpoint (path, or W&B run name) in eval mode; the architecture comes from its saved args."""
    ckpt = torch.load(ckpt if os.path.exists(ckpt) else fetch_model(ckpt), map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = build_ugnn(a["base_channels"], a["gnn_layers"], a["gnn_k"], a.get("cond_channels"),
                       a.get("window", 1))
    model.load_state_dict(ckpt["model"])
    return build_diffusion(model, sampler, sampling_timesteps).to(device).eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name (artifact = <timestamp>_<name>)")
    p.add_argument("--project", default=PROJECT)
    args = p.parse_args()
    path = fetch_model(args.name, args.project)
    if path is None:
        sys.exit(f"no model artifact <timestamp>_{args.name} in {args.project}")
    print(path)
