# -*- coding: utf-8 -*-
"""Pretrain a frozen graph encoder with exact edge-chunked message passing,
then dump H for decoder-only training (potion.train_eval --frozen-h).

Objective: link prediction (positive edges vs random node pairs).
The whole-graph forward runs once per step, in edge chunks, so 123.7M-edge
Amazon2M fits in a few GB instead of ~63 GB.

  python -m potion.pretrain --queries data/amazon2m_queries.pt \
      --steps 500 --out runs/H_amazon2m.pt
"""
import argparse
import os

import numpy as np
import sys
import time

import torch
import torch.nn.functional as F

from .utils import load_queries                      # noqa: E402
from .chunked_encoder import ChunkedSAGE             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--per-step', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--edge-scope', choices=['all', 'train'], default='all',
                    help='all=全部边做链接预测；train=只用两端都在 train 社区的边（更严格，'
                         '避免把测试社区内部结构当训练信号）；最终 H 仍用全图边计算')   
    ap.add_argument('--chunk', type=int, default=2_000_000)
    ap.add_argument('--ckpt', choices=['auto', 'on', 'off'], default='auto',
                    help='梯度检查点：auto=按规模自动；on=强制；off=关闭（快但吃显存）')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    qs, g = load_queries(args.queries)
    n = int(g['num_nodes'])
    feat = torch.as_tensor(g['node_feat'].numpy() if hasattr(g['node_feat'], 'numpy')
                           else g['node_feat'], dtype=torch.float32, device=dev)
    ei = g['edge_index'].long().to(dev)
    ei_fit, neg_pool = ei, None
    if args.edge_scope == 'train':
        tr = [np.asarray(q['target']['community_indices'], dtype=np.int64)
              for q in qs if q.get('split') == 'train']
        train_nodes = np.unique(np.concatenate(tr)) if tr else np.zeros(0, np.int64)
        m = np.isin(ei[0].cpu().numpy(), train_nodes) & np.isin(ei[1].cpu().numpy(), train_nodes)
        ei_fit = ei[:, torch.from_numpy(m).to(dev)]
        neg_pool = torch.from_numpy(train_nodes).to(dev)
        print(f'[edge-scope] 训练边 {ei_fit.size(1)} / 全部 {ei.size(1)}；'
              f'训练节点 {len(train_nodes)} / {n}', flush=True)
    print(f'== graph == n={n} edges={ei.size(1)} feat={feat.size(1)}', flush=True)

    model = ChunkedSAGE(feat.size(1), args.d, args.layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=dev).manual_seed(0)

    for step in range(args.steps):
        t0 = time.time()
        opt.zero_grad()
        H = model(feat, ei_fit, chunk=args.chunk, ckpt=args.ckpt)
        idx = torch.randint(0, ei.size(1), (args.per_step,), device=dev, generator=gen)
        u, v = ei[0, idx], ei[1, idx]
        if neg_pool is None:
            nu = torch.randint(0, n, (args.per_step,), device=dev, generator=gen)
            nv = torch.randint(0, n, (args.per_step,), device=dev, generator=gen)
        else:
            nu = neg_pool[torch.randint(0, neg_pool.numel(), (args.per_step,), device=dev, generator=gen)]
            nv = neg_pool[torch.randint(0, neg_pool.numel(), (args.per_step,), device=dev, generator=gen)]
        pos = (H[u] * H[v]).sum(-1)
        neg = (H[nu] * H[nv]).sum(-1)
        loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + \
            F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        loss.backward()
        opt.step()
        if step % 25 == 0:
            print(f'step {step:4d} loss {loss.item():.4f} ({time.time() - t0:.1f}s)', flush=True)

    model.eval()
    with torch.no_grad():
        H = model(feat, ei, chunk=args.chunk, ckpt=args.ckpt)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    torch.save({'H': H.cpu(), 'n': n, 'd': args.d, 'layers': args.layers,
                'steps': args.steps}, args.out)
    print(f'saved {args.out}  H={tuple(H.shape)}')


if __name__ == '__main__':
    main()
