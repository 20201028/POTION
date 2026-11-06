# POTION: Usage Guide

POTION answers community search (CS) and community detection (CD) queries with a
single set of parameters. A prompt-agnostic encoder reads the graph once, a prompt
encoder turns a query into a token sequence, a bidirectional Transformer decoder
lets prompt tokens and node representations update each other, and two
query-conditioned channels (hop distance and attribute similarity) score every node.

This folder is a self-contained release: every `import` resolves to a module inside
the folder, no script or configuration file outside it is read, and graph data and
query files are passed on the command line. The defaults are the settings used in
the paper.

---

## 1. Layout

```
POTION_release/
├── README.md                 Chinese guide
├── README_EN.md              this file
├── requirements.txt          Python dependencies
├── DataSet/                  raw graphs (.feat/.edges/.circles or Amazon2M .npz/.npy)
├── potion/
│   ├── __init__.py           version and paper defaults (PAPER_DEFAULTS)
│   ├── __main__.py           entry point for `python -m potion`
│   ├── train_eval.py         training + evaluation (was chunk_joint/run.py)
│   ├── model.py              graph encoder / prompt encoder / mask decoder
│   ├── chunked_encoder.py    edge-chunked mean-aggregation encoder (ChunkedSAGE)
│   ├── pretrain.py           link-prediction pretraining, exports H
│   ├── build_queries.py      raw graph -> community-level split -> *_queries.pt
│   ├── gen_amazon2m.py       Amazon2M query builder (adj.npz / feat.npy / labels.npy)
│   ├── utils.py              metrics, losses, query file I/O
│   └── synth.py              synthetic community augmentation
└── examples/toy/             60-node toy graph for a quick smoke test
    ├── toy.feat
    ├── toy.edges
    └── toy.circles
```

---

## 2. Environment

```bash
pip install -r requirements.txt
```

Install the CUDA build of PyTorch from https://pytorch.org first, then the rest.
The encoder is plain PyTorch (edge-chunked mean aggregation); `torch-geometric` is not required.

---

## 3. Data format

Place the raw graph under `<root>/<name>/` as three whitespace-separated text files:

| File | Content per line |
|---|---|
| `<name>.feat` | `node_id  a1 a2 ... aM`, attributes are 0/1 |
| `<name>.edges` | `u v`, one undirected edge (stored both ways internally, no self-loops) |
| `<name>.circles` | `community_id  v1 v2 ...`, one community per line |

See `examples/toy/` for a concrete example. Nodes that never appear in the edge file
are dropped, and communities smaller than `--min-size` (default 16) are discarded.

For very large attribute spaces (e.g., Twitter), compress features when building
queries:

```bash
# keep the 2000 most frequent attributes
python -m potion.build_queries --name twitter --root DataSet --attr-topk 2000 --out DataSet/twitter_queries.pt
# or TF-IDF + SVD down to 128 dimensions (attribute ids are kept)
python -m potion.build_queries --name twitter --root DataSet --attr-embed 128 --out DataSet/twitter_queries.pt
```

---

## 4. Quick smoke test (CPU, about one minute)

```bash
python -m potion.build_queries --name toy --root examples --per-comm 4 --out examples/toy_queries.pt
python -m potion.train_eval --queries examples/toy_queries.pt --epochs 3 --device cpu --M 20 --batch 2
```

The toy graph has only three communities, so the numbers are meaningless; the point
is to confirm that the environment and the pipeline work end to end.

---

## 5. Training and evaluation on real graphs

### 5.1 Build the query file

```bash
python -m potion.build_queries --name <dataset> --root DataSet --out DataSet/<dataset>_queries.pt
```

The split is 6:2:2 at the community level (`--ratio 0.6 0.2 0.2`), each community
yields four queries, and three negative nodes and three negative attributes are
sampled per query (`--neg-mode randk --neg-k 3 --neg-attr-k 3`, seed 4). Test
communities never appear during training. The complete Cora command is in 5.2.

### 5.2 Small graphs (Cora): pretrain H, then train either way

Build queries as in the paper (attributes compressed to 128 dimensions):

```bash
python -m potion.build_queries --name cora --root DataSet \
    --per-comm 4 --min-size 16 --stratify --seed 4 \
    --neg-mode randk --neg-k 3 --neg-attr-k 3 --attr-embed 128 \
    --out DataSet/cora_emb128.pt --force
```

Then pretrain the encoder, export `H`, and run either training mode:

```bash
# pretrain H (link prediction on training-community edges only)
python -m potion.pretrain --queries DataSet/cora_emb128.pt --edge-scope train \
    --steps 500 --out runs/H_cora_train.pt

# stage 1: frozen H, train only the prompt encoder and the decoder
python -m potion.train_eval --queries DataSet/cora_emb128.pt --frozen-h runs/H_cora_train.pt \
    --out runs/cora_frozen

# stage 2: joint training (drop --frozen-h)
python -m potion.train_eval --queries DataSet/cora_emb128.pt --out runs/cora_joint
```

`--cand auto`, synthetic communities (160 / struct / 0.5), `--M 1000`, and
`--batch 8` are already defaults; only `--attr-embed` and `--edge-scope` depend on
the dataset (Amazon2M uses its own builder and keeps the default `--edge-scope all`).

### 5.3 Large graphs (Amazon2M): pretrain H, then freeze the encoder

Amazon2M ships as matrices, so use the dedicated query builder:

```bash
python -m potion.gen_amazon2m --dir DataSet/Amazon2M --per-comm 4 --min-size 16 \
    --stratify --seed 4 --out DataSet/amazon2m_emb100.pt
```

Then pretrain the encoder by link prediction, export `H`, and train in either mode:

```bash
# pretrain the encoder (link prediction) and dump H
python -m potion.pretrain --queries DataSet/amazon2m_emb100.pt --out runs/H_amazon2m_train.pt --steps 500

# stage 1: frozen H, train only the prompt encoder and the decoder
python -m potion.train_eval --queries DataSet/amazon2m_emb100.pt --frozen-h runs/H_amazon2m_train.pt \
    --cand ball --synth-k 0 --synth-per-real 0 --out runs/amazon2m

# stage 2: joint training (drop --frozen-h)
python -m potion.train_eval --queries DataSet/amazon2m_emb100.pt \
    --cand ball --synth-k 0 --synth-per-real 0 --out runs/amazon2m_joint
```

**Pass `--cand ball` explicitly.** Training coverage on Amazon2M is only 0.169, below
the 0.85 threshold of `--cand auto`, so auto falls back to full-graph decoding and
runs out of memory at 2.45M nodes. Synthetic communities are also disabled on
Amazon2M (`--synth-k 0`), matching the actual runs. Both stages use the same
`potion.train_eval` entry point and differ only by `--frozen-h`.

Peak memory of `pretrain.py` is controlled by `--chunk` (2,000,000 edges per chunk by
default); `--ckpt auto` enables gradient checkpointing automatically. Amazon2M peaks
at about 7 GB.
---

## 6. Defaults (the paper settings)

The argparse defaults in `train_eval.py` are already the values used in the paper, so
normally no flag is needed:

| Symbol | Flag | Default |
|---|---|---|
| $d$ | `--d` | 128 |
| encoder layers | `--layers` | 3 |
| decoder | fixed in code | 3 layers, 4 heads |
| batch size | `--batch` | 8 |
| epochs | `--epochs` | 200 |
| learning rate / weight decay | `--lr` / `--weight-decay` | 1e-3 / 1e-4 |
| early-stopping patience | `--patience` | 30 |
| random seeds | `--seeds` | 0 1 2 |
| interaction rounds $R$ | `--rounds` | 2 |
| candidate radius $r_s$ | `--cand-hop` | 2 |
| candidate cap $c_{\max}$ | `--cand-cap` | 15000 |
| candidate strategy | `--cand` | auto (local if coverage suffices, else full) |
| sampled nodes $M_s$ | `--M` | 1000 |
| synthetic communities $K_{\mathrm{syn}}$ | `--synth-k` | 160 |
| synthetic ratio $\rho$ | `--synth-per-real` | 0.5 |
| synthetic size cap | `--synth-attr-max` | 64 |
| negative-bearing share $\nu$ | `--neg-cover` | 2/3 |
| attribute channel strength $w$ | `--attr-w` | 1.0 |
| BCE weight cap $w_{\max}$ | `--pw-max` | 50 |
| Dice weight $\lambda$ | fixed in code | 1 |
| hop buckets $\beta$ | fixed in code | 5 |
| graph encoder | fixed in code | 3-layer mean aggregation (ChunkedSAGE, no self-loops) |
| prompt token source | `--prompt-src` | H (encoder output) |
| position channel | `--pos-mode` / `--pos-chan` | feat / hop |
| attribute channel | `--attr-mode` | sim |
| prompt category set | `--prompt-cats` | taxonomy (8 forms) |

The toy commands above override `--epochs`, `--M`, and `--batch` only to finish fast.

---

## 7. Output

| Item | Description |
|---|---|
| stdout | per-epoch validation F1, early-stop messages, retrieval table over the eight prompt forms, detection AvgF1 |
| `--out` directory | written only with `--save`: `best.pt` and `config.json` (all arguments) |
| exit code | 0 on success; out-of-memory conditions are reported in the log |

With multiple seeds (`--seeds 0 1 2`) results are summarised as mean +/- std.

---

## 8. FAQ

**Out of memory.** Lower `--batch` on small graphs. On large graphs, use
`potion.pretrain` with a smaller `--chunk` first, then train with `--frozen-h`.
`--cand-cap` bounds the candidate set of a single decode and also saves memory.

**Training is slow.** `--cand auto` already switches between local and full decoding
by coverage; use `--epochs 50` to check the pipeline first. CPU is fine for the toy
graph; real graphs need a GPU.

**Reproducing a single run.** Fix `--seeds` (the default 0 1 2 is fixed) and
`--device`; `build_queries.py` uses `--seed` (default 0) for the community split and
sampling. With both fixed, results are reproducible.

**Using another dataset.** Put the three files in the format of Section 3, change
`--name` and `--root`, and nothing else: community count and attribute dimension are
read from the files.
