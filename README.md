# Graph diffusion for long-horizon link scheduling

A δ-conditioned graph diffusion model (**U-GNN**) that generates wireless link schedules on a
conflict graph under a per-link **minimum-rate constraint** δ. The model is trained to imitate a
teacher, **DGD**: maximum-weight independent sets (solved with KaMIS) under dual-dynamics link
weights ([arXiv:2607.18480](https://arxiv.org/abs/2607.18480), eq. 21). It is compared against
that teacher and against **SAGNN**, a state-augmented GNN policy.

One conditioned model covers every δ: it is trained at δ ∈ {0.01, 0.05, 0.1} (fine-tuned with
δ = 0.15 added for the δ-sweep figure), evaluated at every δ from 0.01 to 0.20, and transferred
zero-shot from 500-link graphs to graphs with up to 5000 links.

## Repository layout

```
src/
  ugnn/                   U-GNN: dataset, model (DDPM train / DDIM sample), train, generate, metrics
  sagnn/                  SAGNN baseline + conflict-graph generation (datageneration, make_graphs, make_split)
  MIS/                    DGD teacher: file-based MWIS trajectory pipeline around KaMIS
    run_pipeline.sh         one T-slot trajectory per graph (KaMIS.py does init / solve / weight update)
    KaMIS.py
  common/                 shared utilities (conflict graphs, interference, parallel timing pool)
  graph_signal_diffusion/ trimmed vendored copy: the U-GNN denoiser, DDPM/DDIM samplers and their configs
scripts/                  experiment drivers (see "Reproducing the paper")
notebooks/
  paper_plots2.ipynb      every paper figure, from the cached results
  paper_figs2/            the rendered figures (PDF)
```

Datasets, checkpoints and rollout caches are **not** in the repository; the scripts regenerate
them under `data/` (gitignored).

## Setup

```bash
conda create -n gdls python=3.12 && conda activate gdls
pip install torch                       # pick the build for your CUDA version
pip install -r requirements.txt         # torch-geometric, hydra, wandb, ...
```

Tested with Python 3.12, PyTorch 2.8 (CUDA 12.8) and PyTorch Geometric 2.6.1. Training and U-GNN
rollouts expect a GPU; the DGD teacher runs on CPU (one KaMIS process per core).

**KaMIS** (the MWIS solver behind DGD) is built from source into `src/MIS/external/`:

```bash
git clone https://github.com/KarlsruheMIS/KaMIS.git src/MIS/external/KaMIS
cd src/MIS/external/KaMIS && ./compile_withcmake.sh && cd -
```

The pipeline calls `deploy/sort_adjacencies` and `deploy/weighted_local_search`. The results here
used KaMIS commit `7f3e257`.

All Python entry points run from `src/` with `PYTHONPATH=.`; the shell scripts set this themselves.

**Weights & Biases.** `ugnn.train` logs to W&B and uploads the final checkpoint as a model artifact
(project `ugnn_link_scheduling`). Use `--no-wandb --keep-local-model` to train fully offline.
Scripts that look a model up by run name (`python -m ugnn.model --name ...`) need a W&B login
with access to the project that holds it; point them at local checkpoints otherwise.

## Reproducing the paper

The base set is 150 irregular conflict graphs (100 train / 50 test, 256–324 users, ~400–500
links), stored as `data/sawl_dataset_irregular_lcc_degree6_noise0.02_graphs_150_nodes256-324.pt`.

### 1. Conflict graphs

- **Base set:** `sagnn.datageneration.create_conflict_dataset_irregular(...)` with noise 0.02,
  `lcc=True`, radius 1.2, target conflict degree 6 (tolerance 0.5).
- **Size-transfer families** (`data/irregular_links{1000,2000,5000}/`), matched in conflict degree:
  ```bash
  cd src && PYTHONPATH=. python -m sagnn.make_graphs --world-sizes <w> ... --noise <noise> \
      --n-train 100 --n-test 50 --sparse --out-dir ../data/irregular_links1000
  ```

### 2. DGD teacher trajectories

Every training graph is rolled at every training δ (dual rule, η = 2, weight scale 1000, T = 100),
so δ is the only thing that separates the targets for the same graph:

```bash
# run from the repository root; the script cd's into src/, so pass absolute paths
BASE=$PWD/data/sawl_dataset_irregular_lcc_degree6_noise0.02_graphs_150_nodes256-324.pt
OUT=$PWD/data/delta_grid/irregular_lcc_noise0.02

# training set -> $OUT/train_T100_grid.pt
DS=$BASE OUT_ROOT=$OUT T=100 DELTAS="0.01 0.05 0.1" ./scripts/collect_delta_grid.sh

# test trajectories at every evaluated δ
DS=$BASE OUT_ROOT=$OUT T=100 SPLIT=test N_GRAPHS=50 \
  DELTAS="0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2" \
  ./scripts/collect_delta_grid.sh
```

### 3. Train U-GNN (500-link model)

```bash
cd src && PYTHONPATH=. python -m ugnn.train \
  --data ../data/delta_grid/irregular_lcc_noise0.02/train_T100_grid.pt --name ugnn_w10_bc512 \
  --results-dir ../data/delta_grid/irregular_lcc_noise0.02/runs \
  --window 10 --stride 1 --windows 10 --epochs 1000 --batch-size 60 --accum 2 --lr 1e-4 \
  --base-channels 512 --gnn-layers 2 --gnn-k 2 --save-every-n-epochs 100 --keep-local-model
```

Each example is a 10-slot window of a teacher trajectory, conditioned on its δ; 10 windows are
drawn per trajectory. Sampling uses 10 DDIM steps under bfloat16.

The δ-sweep figure uses this model fine-tuned for 300 more epochs with δ = 0.15 added:
`scripts/run_finetune_delta015.sh` (teacher at δ = 0.15, merged training set, fine-tune, evaluation
via `scripts/eval_finetune_delta015.py`).

### 4. SAGNN baseline

```bash
./scripts/run_sagnn_irregular500.sh   # train the policy, dual-roll it on the 500-link test set per δ
./scripts/run_sagnn_transfer.sh       # same policy, zero-shot on the 1000/2000/5000-link families
```

### 5. Size transfer

```bash
./scripts/run_size_transfer_irregular.sh   # teacher per size, native models, 4x4 transfer matrix
./scripts/run_extend_early_stop.sh         # extend native 2000/5000-link models (early_stop_check.py)
./scripts/run_extend_2000_to_1000.sh       # extend the 2000-link model to 1000 epochs
./scripts/extend_native_training.sh        # bring native models up to the 500-link model's budget
python scripts/resample_transfer_matrix.py # re-roll the matrix with the paper's sampler
python scripts/bench_inference.py          # per-schedule inference cost: DGD (CPU) vs U-GNN / SAGNN (GPU)
```

`scripts/transfer_row.py` rolls one model on every size family and swaps it into the matrix.

### 6. Figures

Open `notebooks/paper_plots2.ipynb` from `notebooks/` and run it top to bottom. U-GNN rollouts
are cached under `data/delta_grid/irregular_lcc_noise0.02/results/`, so re-runs are fast. Scoring
is strict: two links that collide in a slot both fail.

## Credits

- `src/graph_signal_diffusion/` is a trimmed copy of the Graph Signal Generative Diffusion
  codebase (U-GNN denoiser, DDPM/DDIM).
- `src/MIS/KaMIS.py` and `run_pipeline.sh` adapt Romi's file-based MWIS
  trajectory pipeline.
- MWIS solving uses [KaMIS](https://github.com/KarlsruheMIS/KaMIS) (Lamm, Sanders, Schulz,
  Strash, Werneck; MIT licence), which is cloned at setup and not redistributed here.

## License

MIT; see [LICENSE](LICENSE). The vendored and adapted components keep their authors' credit above.
