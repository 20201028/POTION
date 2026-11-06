# -*- coding: utf-8 -*-
"""从 Amazon2M 生成 CAM 查询文件（v2 格式）。

Amazon2M 原始文件（Amazon2M/ 下）：
  Amazon2M_adj.npz     scipy CSR（indptr/indices/data），对称，1.237 亿非零
  Amazon2M_feat.npy    (n, 100) float64 稠密节点特征
  Amazon2M_labels.npy  (n, 47) float32 one-hot 真值社区

离散属性按 Huang & Lakshmanan, PVLDB 2017（[20]）的实验协议合成：
  · 属性池 |A| = 0.005 × |V|（Amazon2M 上 = 12,245）
  · 每个真值社区随机选 3 个属性，每个属性分配给该社区随机 80% 的节点
  · 每个节点再从整个属性池随机加 1–5 个属性（噪声）
  · 全部二值
feat 只有 100 维（≤128），按约定**直接当节点特征**，不再做 SVD 压缩。

输出与 sampleQuerySplit.py 完全相同的 .pt 结构：
  graph: edge_index / num_nodes / num_attributes(=属性池大小) / attr_dim(=100) /
         attr_ptr / attr_idx / node_feat / communities / community_attributes
  query_batchs: [[query, ...]]

用法：
  python -m potion.gen_amazon2m --per-comm 4 --stratify --seed 4 \
      --out DataSet/amazon2m_csplit4_strat4_queries.pt
  # 先小规模冒烟：
  python -m potion.gen_amazon2m --limit-nodes 200000 --per-comm 2 --out temp/amazon_smoke.pt
"""
import argparse
import os
import random

import numpy as np
import torch

from .build_queries import stratified_split
from .utils import struct_features


def build_graph(args):
    d = args.dir
    z = np.load(os.path.join(d, 'Amazon2M_adj.npz'), allow_pickle=True)
    indptr = np.asarray(z['indptr'], dtype=np.int64)
    indices = np.asarray(z['indices'], dtype=np.int64)
    n_all = int(z['shape'][0])
    feat = np.load(os.path.join(d, 'Amazon2M_feat.npy'), mmap_mode='r')
    labels = np.load(os.path.join(d, 'Amazon2M_labels.npy'), mmap_mode='r')

    n = min(args.limit_nodes, n_all) if args.limit_nodes else n_all
    print(f'== Amazon2M ==  节点 {n_all}（使用 {n}）  边(有向) {indices.size}  特征 {feat.shape[1]} 维')

    # ---- 边索引（CSR 本身对称；limit 时只保留两端都在前 n 的边）----
    deg = np.diff(indptr)
    src = np.repeat(np.arange(len(deg), dtype=np.int64), deg)
    if n < n_all:
        keep = (src < n) & (indices < n)
        src, dst = src[keep], indices[keep]
        deg_n = np.bincount(src, minlength=n)
    else:
        dst = indices
        deg_n = deg
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).contiguous()
    print(f'   保留边 {edge_index.size(1)}（有向），平均度 {edge_index.size(1) / n:.1f}')

    # ---- 社区：one-hot 标签取 argmax（每题一个真值社区）----
    lab = np.asarray(labels[:n])
    cls = lab.argmax(axis=1).astype(np.int64)
    communities = {}
    for c in np.unique(cls):
        members = np.nonzero(cls == c)[0]
        if len(members) >= args.min_size:
            communities[int(c)] = members
    print(f'   真值社区 {len(communities)} 个（min-size {args.min_size}），'
          f'规模 {min(len(v) for v in communities.values())}–{max(len(v) for v in communities.values())}')

    # ---- 节点特征：feat(100) ⊕ 5 维结构（不再 SVD）----
    x = torch.from_numpy(np.asarray(feat[:n], dtype=np.float32))
    node_feat = torch.cat([x, struct_features(edge_index, n)], dim=1)
    print(f'   node_feat {node_feat.size(1)} 维（{x.size(1)} 维原始特征 ⊕ 5 维结构）')

    # ---- 离散属性：优先读取预合成 CSR（gen_amazon2m_attrs.py 的产物）----
    if args.attrs_from and os.path.exists(args.attrs_from):
        z = np.load(args.attrs_from, allow_pickle=True)
        m = int(z['num_attributes'][0])
        indptr_all = np.asarray(z['indptr'], dtype=np.int64)
        idx_all = np.asarray(z['indices'], dtype=np.int64)
        if n < n_all:                      # 只保留前 n 个节点的属性
            attr_ptr = indptr_all[:n + 1].copy()
            attr_idx = idx_all[indptr_all[0]:indptr_all[n]].copy()
            attr_ptr -= attr_ptr[0]
        else:
            attr_ptr, attr_idx = indptr_all, idx_all
        comm_attrs = {int(c): [int(a) for a in row]
                      for c, row in enumerate(np.asarray(z['community_attrs']))}
        print(f'   属性来自 {args.attrs_from}：池 {m} 维；nnz {len(attr_idx)}'
              f'（每节点 {len(attr_idx) / n:.2f} 个）')
    else:
        # 与 gen_amazon2m_attrs.py 完全一致：同一批 80% 成员共享 3 个属性
        m = int(args.attr_pool_ratio * n_all)
        rng = np.random.default_rng(args.attr_seed)
        rows, cols = [], []
        comm_attrs = {}
        for cid, members in communities.items():
            mem = np.asarray(members, dtype=np.int64)
            picked = rng.choice(m, size=args.attrs_per_comm, replace=False)
            comm_attrs[cid] = [int(a) for a in picked]
            cnt = max(1, int(args.attr_cover * len(mem)))
            sel = rng.choice(mem, size=cnt, replace=False)
            for a in picked:
                rows.append(sel)
                cols.append(np.full(cnt, int(a), dtype=np.int64))
        k = rng.integers(args.noise_min, args.noise_max + 1, size=n)
        rows.append(np.repeat(np.arange(n, dtype=np.int64), k))
        cols.append(rng.integers(0, m, size=int(k.sum())))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        key = rows * np.int64(m) + cols
        key = np.unique(key)
        rows, cols = key // m, key % m
        order = np.argsort(rows, kind='stable')
        rows, cols = rows[order], cols[order]
        attr_ptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))]).astype(np.int64)
        attr_idx = cols.astype(np.int64)
        print(f'   合成属性池 {m} 维；nnz {len(attr_idx)}（每节点 {len(attr_idx) / n:.2f} 个）')

    graph = {
        'edge_index': edge_index,
        'num_nodes': n,
        'num_attributes': m,
        'attr_dim': int(x.size(1)),          # 稠密块宽度（= attr_emb 的列数）
        'attr_emb': x,                        # 与 Cora/Twitter 的 attr_emb 同义：聚类用的稠密表示
        'attr_ptr': torch.from_numpy(attr_ptr),
        'attr_idx': torch.from_numpy(attr_idx),
        'node_feat': node_feat,
        'communities': {int(c): members.tolist() for c, members in communities.items()},
        'community_attributes': {
            int(c): {int(a): 1 for a in attrs} for c, attrs in comm_attrs.items()},
    }
    return graph, communities, comm_attrs, attr_ptr, attr_idx, n, m


def build_queries(args, graph, communities, comm_attrs, attr_ptr, attr_idx, n, m):
    rng = random.Random(args.seed)
    cids = sorted(communities)
    sizes = {c: len(communities[c]) for c in cids}
    n_tr = int(len(cids) * args.ratio[0])
    n_va = int(len(cids) * args.ratio[1])
    targets = {'train': n_tr, 'val': n_va, 'test': max(0, len(cids) - n_tr - n_va)}
    order = sorted(cids, key=lambda c: sizes[c])
    if args.stratify:
        split_of = stratified_split(order, targets)
    else:
        rng.shuffle(order)
        split_of = {c: ('train' if i < n_tr else ('val' if i < n_tr + n_va else 'test'))
                    for i, c in enumerate(order)}
    cids = order

    indptr = graph['edge_index'][0]        # 仅用于打印，邻接查表用 numpy
    ei = graph['edge_index'].numpy()
    src, dst = ei[0], ei[1]
    # 邻接：把 CSR 还原成每行邻居切片的指针（原图邻接就是对称 CSR）
    z = np.load(os.path.join(args.dir, 'Amazon2M_adj.npz'), allow_pickle=True)
    indptr, indices = np.asarray(z['indptr'], np.int64), np.asarray(z['indices'], np.int64)

    def neighbors(v):
        return indices[indptr[v]:indptr[v + 1]]

    queries, n_neg_nonzero = [], 0
    for cid in cids:
        members = communities[cid]
        member_set = set(members.tolist())
        for _ in range(args.per_comm):
            k_seed = rng.randint(1, min(3, len(members)))
            seeds = rng.sample(members.tolist(), k_seed)
            # 负点：种子的邻居里、不属于目标社区的节点（B\A）
            cand = set()
            for s in seeds:
                cand.update(int(v) for v in neighbors(int(s)))
            cand -= member_set
            neg = rng.sample(sorted(cand), min(args.neg_k, len(cand))) if cand else []
            n_neg_nonzero += int(bool(neg))
            # 正属性：该社区的 3 个合成属性里随机取 2 个；负属性：池中随机取，避开正属性
            pos_attrs = rng.sample(comm_attrs[cid], min(args.pos_attr_k, len(comm_attrs[cid])))
            neg_pool = [a for a in rng.sample(range(m), min(m, 3 * len(pos_attrs) + 10))
                        if a not in comm_attrs[cid]]
            neg_attrs = rng.sample(neg_pool, min(args.neg_attr_k, len(neg_pool))) if neg_pool else []
            mask = torch.zeros(n, dtype=torch.bool)
            mask[torch.tensor(members.tolist(), dtype=torch.long)] = True
            queries.append({
                'query': {
                    'nodes': torch.tensor(seeds, dtype=torch.long),
                    'attrs': torch.tensor(pos_attrs, dtype=torch.long),
                    'neg_nodes': torch.tensor(neg, dtype=torch.long),
                    'neg_attrs': torch.tensor(neg_attrs, dtype=torch.long),
                    'neg_mode': 'randk',
                },
                'target': {
                    'community_id': int(cid),
                    'community_indices': members.tolist(),
                    'community_mask': mask,
                },
                'split': split_of[cid],
            })
    print(f'   查询 {len(queries)} 条（每社区 {args.per_comm}）；含负点的 {n_neg_nonzero} 条')
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='DataSet/Amazon2M')
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-comm', type=int, default=4)
    ap.add_argument('--min-size', type=int, default=16)
    ap.add_argument('--ratio', type=float, nargs=3, default=[0.6, 0.2, 0.2])
    ap.add_argument('--seed', type=int, default=4)
    ap.add_argument('--stratify', action='store_true')
    ap.add_argument('--pos-attr-k', type=int, default=3, help='每条查询的正属性个数上限（与 sampleQuerySplit 一致）')
    ap.add_argument('--neg-k', type=int, default=3)
    ap.add_argument('--neg-attr-k', type=int, default=3)
    ap.add_argument('--limit-nodes', type=int, default=0, help='只保留前 N 个节点（冒烟用）')
    # [20] 协议参数
    ap.add_argument('--attr-seed', type=int, default=42,
                    help='现场合成属性时的随机种子（与 gen_amazon2m_attrs.py 一致）')
    ap.add_argument('--attrs-from', default='DataSet/Amazon2M/Amazon2M_attrs.npz',
                    help='直接加载预合成的属性 CSR（默认读取 gen_amazon2m_attrs.py 的产物）；传空串则现场合成')
    ap.add_argument('--attr-pool-ratio', type=float, default=0.005)
    ap.add_argument('--attrs-per-comm', type=int, default=3)
    ap.add_argument('--attr-cover', type=float, default=0.8)
    ap.add_argument('--noise-min', type=int, default=1)
    ap.add_argument('--noise-max', type=int, default=5)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    graph, communities, comm_attrs, attr_ptr, attr_idx, n, m = build_graph(args)
    queries = build_queries(args, graph, communities, comm_attrs, attr_ptr, attr_idx, n, m)

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    torch.save({'graph': graph, 'query_batchs': [queries],
                'metadata': {'name': 'amazon2m', 'per_comm': args.per_comm,
                             'nodes': n, 'attr_pool': m,
                             'splits': {s: sum(1 for q in queries if q['split'] == s)
                                        for s in ('train', 'val', 'test')}}}, out)
    print(f'已保存 {out}（{os.path.getsize(out) / 1e9:.2f} GB）')


if __name__ == '__main__':
    main()
