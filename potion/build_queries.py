# -*- coding: utf-8 -*-
"""社区级划分的查询生成（自带图读取，不再依赖 sampleQuery.py）：
先把社区分成 train/val/test 三份，再从每个社区生成指定数量的查询。
正点/正属性采样规则与旧版 sampleQuery 一致；
**负字段（负点/负属性）另有口径**，见 `--neg-mode`。

与旧版（查询级划分）的区别：
  旧版     查询轮转生成后按查询划分（train/test 社区池共享）
  本文件   社区级：社区先三份划分，测试社区在训练中零监督
           （消除社区池泄露，测"未见社区"的定位能力）
           ⚠ 负字段采样已参数化（rand1/randk/edgek），两边**不再**逐位等价
             ——用哪个文件生成的 .pt 要记清。

输出的每条查询带 split 标记；run.py 检测到后按社区级划分使用。

用法：python -m potion.build_queries --name facebook --per-comm 8 \
          --out DataSet/facebook_csplit_queries.pt

阶段32（负信息剂量/位置）：
  # 0) 先自检：rand1 应与现有 .pt 完全一致（随机流纪律的机器证明）
  python -m potion.build_queries --name facebook --per-comm 20 --min-size 16 --stratify \
      --seed 4 --neg-mode rand1 --out temp/_repro_rand1.pt
  python temp/cmp_pt.py --a DataSet/facebook_csplit_strat4_queries.pt --b temp/_repro_rand1.pt

  # 1) 只加剂量 / 剂量+位置（新文件，绝不覆盖基线）
  python -m potion.build_queries ... --neg-mode randk --neg-k 10 \
      --out DataSet/facebook_csplit_strat4_queries_randk10.pt
  python -m potion.build_queries ... --neg-mode edgek --neg-k 10 \
      --out DataSet/facebook_csplit_strat4_queries_edgek10.pt

  # 2) 数据侧验收（②③⑥ 必须不变、①④⑤ 必须变）
  python temp/audit_negs.py DataSet/facebook_csplit_strat4_queries_edgek10.pt
"""
import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch

from .utils import build_node_feat, struct_features


class CommunityQueryLoader:
    """从 DataSet/<name>/ 读图与社区（原 sampleQuery.py 的最小自包含实现）。

    只保留本文件需要的部分：`.edges` / `.feat` / `.circles` → 图字典。
    与原实现的两点区别（读取口径其余逐行一致，所以生成的查询文件与旧版可比）：
      · **不再构造稠密 adj**（n×n，facebook 3959²×4 ≈ 63 MB）——下游用 edge_index 重建；
      · **不再读 `*_node_emb.txt` / `*_emb.npy`**——原实现只把结果塞进
        `graph['node_embeddings']`，而新 .pt 的白名单不再保存它们。
    """

    def __init__(self, root, dataset, min_community_size, min_community_num, attr_topk=0,
                 attr_embed=0, attr_cache=True, attr_cache_path=None):
        self.root = root
        self.dataset = dataset
        self.min_community_size = min_community_size
        self.min_community_num = min_community_num
        # 0 = 保留全部属性；>0 = 只保留文档频率最高的 topk 列（见 _load_features_topk）。
        # twitter 这类 (n, m) 稠密特征矩阵可达几十 GB，不上裁剪连 GPU 都放不下。
        self.attr_topk = int(attr_topk or 0)
        # >0 = 用 TF-IDF + SVD 把属性压成 k 维嵌入（见 _load_features_embed），
        # .pt 里不再有稠密 A，只存 attr_emb(n×k) + attr_ptr/attr_idx(CSR)。
        self.attr_embed = int(attr_embed or 0)
        # 属性嵌入缓存：同一图重复生成查询时复用 SVD 结果（--no-attr-cache 可关）
        self.attr_cache = bool(attr_cache)
        self.attr_cache_path = attr_cache_path

    def _nodes_in_edges(self, edge_file):
        """边文件里出现过的原始节点 id 集合（用于在加载特征时就排除孤立点）。"""
        if not os.path.isfile(edge_file):
            return None
        nodes = set()
        with open(edge_file, encoding='utf-8') as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 2:
                    nodes.add(int(p[0]))
                    nodes.add(int(p[1]))
        return nodes

    def _load_features(self, file_path, keep_nodes=None):
        """读 `.feat`（每行：节点 id + 0/1 向量）。

        返回 (nodes_feats, node_feature_indices, node_in_dim, n_nodes, id_mapping)。
        节点顺序 = 原始 id 升序（只含"出现在边文件里且至少一个属性为 1"的节点），
        与原实现的 id_mapping 完全一致 —— 换顺序会让已有 .pt 的节点编号错位。
        """
        if not os.path.isfile(file_path):
            raise Exception(f"No such file: {file_path}")
        feats_node = {}
        node_feats_nid = {}
        unique_nodes = set()
        node_in_dim = 0
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                values = list(map(int, line.strip().split()))
                if len(values) < 2:
                    continue
                feature_vector = values[1:]
                node_in_dim = len(feature_vector)
                feature_indices = [idx for idx, val in enumerate(feature_vector) if val == 1]
                if len(feature_indices) == 0:
                    continue
                node_id = values[0]
                if keep_nodes is not None and node_id not in keep_nodes:
                    continue
                if node_id in unique_nodes:
                    continue
                unique_nodes.add(node_id)
                node_feats_nid[node_id] = feature_indices
                feats_node[node_id] = np.array(feature_vector, dtype=np.float32)

        original_nodes = sorted(unique_nodes)
        id_mapping = {orig: new for new, orig in enumerate(original_nodes)}
        reverse_mapping = {new: orig for orig, new in id_mapping.items()}
        n_nodes = len(original_nodes)
        nodes_feats = np.array([feats_node[reverse_mapping[i]] for i in range(n_nodes)])
        node_feature_indices = {i: node_feats_nid[reverse_mapping[i]] for i in range(n_nodes)}
        print(f"Loaded dense features: {len(node_feats_nid)} nodes, "
              f"feature dim: {node_in_dim}, matrix shape: {nodes_feats.shape}")
        return nodes_feats, node_feature_indices, node_in_dim, n_nodes, id_mapping

    def _load_features_topk(self, file_path, keep_nodes, topk):
        """流式读 `.feat`，只保留文档频率最高的 topk 个属性。

        为什么需要：`node_feat` 是 n×m 的**稠密**矩阵，twitter 上可达几十 GB，
        连 GPU 都放不下（`run.py` 会 OOM）。这里两遍流式读取，永远不构造完整矩阵：
          pass 1：统计每列文档频率 + 收集有效节点集合（口径与 `_load_features` 一致：
                  完整向量里有 ≥1 个 1，且出现在边文件里）；
          pass 2：只为保留列分配 n×topk 的矩阵并填充。
        节点集合、id_mapping 与不裁剪时**完全相同**（排序、去重、过滤规则一致），
        所以边/社区/查询划分都不变——只是属性维度变窄。
        """
        if not os.path.isfile(file_path):
            raise Exception(f"No such file: {file_path}")
        df = None
        valid = set()
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                vals = list(map(int, line.strip().split()))
                if len(vals) < 2:
                    continue
                vec = vals[1:]
                arr = np.asarray(vec, dtype=np.int8)
                if not arr.any():
                    continue
                node_id = vals[0]
                if keep_nodes is not None and node_id not in keep_nodes:
                    continue
                if node_id in valid:
                    continue
                valid.add(node_id)
                if df is None:
                    df = np.zeros(len(arr), dtype=np.int64)
                elif len(arr) > len(df):
                    df = np.pad(df, (0, len(arr) - len(df)))
                df[:len(arr)] += arr
        if df is None:
            raise Exception(f'特征文件为空或全零：{file_path}')
        n_all = len(df)
        keep = np.sort(np.argsort(-df, kind='stable')[:topk]) if n_all > topk else np.arange(n_all)
        keep_set = set(int(a) for a in keep)
        remap = {int(o): i for i, o in enumerate(keep)}
        print(f'[属性裁剪] {n_all} → {len(keep)} 维（按文档频率取 top-{topk}）')

        original_nodes = sorted(valid)
        id_mapping = {orig: new for new, orig in enumerate(original_nodes)}
        n_nodes = len(original_nodes)
        nodes_feats = np.zeros((n_nodes, len(keep)), dtype=np.float32)
        node_feats_nid = {}
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                vals = list(map(int, line.strip().split()))
                if len(vals) < 2:
                    continue
                node_id = vals[0]
                if node_id not in id_mapping:
                    continue
                vec = vals[1:]
                arr = np.asarray(vec, dtype=np.int8)
                idxs = [remap[int(i)] for i in np.nonzero(arr)[0] if int(i) in keep_set]
                row = id_mapping[node_id]
                nodes_feats[row, idxs] = 1.0
                node_feats_nid[row] = idxs
        print(f"Loaded dense features (top-{len(keep)}): {n_nodes} nodes, "
              f"matrix shape: {nodes_feats.shape}")
        return nodes_feats, node_feats_nid, len(keep), n_nodes, id_mapping

    def _load_features_embed(self, file_path, keep_nodes, k, cache_path=None):
        """流式读 `.feat` → TF-IDF + 随机化 SVD 的 k 维属性嵌入（不构造稠密 A）。

        返回 (emb, node_feats_nid, num_attributes, n_nodes, id_mapping, attr_ptr, attr_idx)：
          emb        (n, k) float32，节点属性嵌入（U·S）
          attr_ptr   (n+1,) CSR 行指针
          attr_idx   (nnz,) 属性 id（保留原始 id 空间，提示 token 仍用它）
        节点集合、id_mapping 与不压缩时完全一致；属性 id 也保持原值（num_attributes=m），
        所以属性通道可以用 attr_ptr/attr_idx 精确计算，不受嵌入影响。
        """
        if not os.path.isfile(file_path):
            raise Exception(f"No such file: {file_path}")
        df = None
        valid = set()
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                vals = list(map(int, line.strip().split()))
                if len(vals) < 2:
                    continue
                arr = np.asarray(vals[1:], dtype=np.int8)
                if not arr.any():
                    continue
                node_id = vals[0]
                if keep_nodes is not None and node_id not in keep_nodes:
                    continue
                if node_id in valid:
                    continue
                valid.add(node_id)
                if df is None:
                    df = np.zeros(len(arr), dtype=np.int64)
                elif len(arr) > len(df):
                    df = np.pad(df, (0, len(arr) - len(df)))
                df[:len(arr)] += arr

        original_nodes = sorted(valid)
        id_mapping = {orig: new for new, orig in enumerate(original_nodes)}
        n_nodes = len(original_nodes)
        m = 0 if df is None else len(df)
        rows, cols = [], []
        node_feats_nid = {}
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                vals = list(map(int, line.strip().split()))
                if len(vals) < 2:
                    continue
                node_id = vals[0]
                if node_id not in id_mapping:
                    continue
                nz = np.nonzero(np.asarray(vals[1:], dtype=np.int8))[0]
                r = id_mapping[node_id]
                node_feats_nid[r] = [int(a) for a in nz]
                rows.append(np.full(len(nz), r, dtype=np.int64))
                cols.append(nz.astype(np.int64))

        import scipy.sparse as sp
        if m < 2 or n_nodes < 2:
            raise Exception(f'属性维度过小（m={m}, n={n_nodes}），无法做 SVD；请改用 --attr-topk')
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(n_nodes, m))
        # TF-IDF：idf = log((1+n)/(1+df)) + 1，再逐行 L2
        dfc = np.bincount(cols, minlength=m).astype(np.float32)
        idf = np.log((1.0 + n_nodes) / (1.0 + dfc)) + 1.0
        A = A.multiply(idf[np.newaxis, :]).tocsr()
        norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel()) + 1e-9
        A = sp.diags((1.0 / norms).astype(np.float32)) @ A
        A = A.tocsr()
        kk = int(min(k, min(A.shape) - 1))

        # ---- 嵌入缓存：同一 (节点集, m, k) 直接复用 ----
        # SVD 是这条链上最慢的一步；换 per-comm/seed/neg-mode 重新生成查询时，节点集、
        # 属性维度、id_mapping 都不变 ⇒ 嵌入完全相同，没必要重算。
        nodes_arr = np.asarray(original_nodes, dtype=np.int64)
        emb = None
        if cache_path and os.path.isfile(cache_path):
            try:
                z = np.load(cache_path, allow_pickle=False)
                if (int(z['k']) == kk and int(z['m']) == m
                        and z['nodes'].shape == nodes_arr.shape
                        and np.array_equal(z['nodes'], nodes_arr)):
                    emb = z['emb'].astype(np.float32)
                    print(f'[属性嵌入] 复用缓存 {cache_path}（{m} → {kk} 维）')
                else:
                    print(f'[属性嵌入] 缓存与当前图不匹配，重新计算（{cache_path}）')
            except Exception as e:
                print(f'[属性嵌入] 缓存读取失败（{e}），重新计算')
        if emb is None:
            from scipy.sparse.linalg import svds
            try:
                U, S, _ = svds(A, k=kk, random_state=0)
            except TypeError:                   # 老版本 scipy 没有 random_state
                U, S, _ = svds(A, k=kk)
            emb = (U * S).astype(np.float32)
            print(f'[属性嵌入] {m} 维 → {kk} 维（TF-IDF + SVD）；'
                  f'nnz {len(rows)}，保留原始属性 id 供属性通道使用')
            if cache_path:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
                np.savez(cache_path, emb=emb, nodes=nodes_arr,
                         k=np.int64(kk), m=np.int64(m))
                print(f'[属性嵌入] 已缓存 → {cache_path}')
        return (emb, node_feats_nid, m, n_nodes, id_mapping,
                A.indptr.astype(np.int64), A.indices.astype(np.int64))

    def _build_edge_index(self, edge_file, id_mapping):
        """读 `.edges` → PyG 风格 (2, 2E) 边索引（无向边存两次、无自环、无重复）。"""
        edges, edge_str = [], set()
        for line in open(edge_file, encoding='utf-8'):
            node1, node2 = line.strip().split()
            a, b = int(node1), int(node2)
            key = f"{min(a, b)}_{max(a, b)}"
            if a == b or key in edge_str or a not in id_mapping or b not in id_mapping:
                continue
            edge_str.add(key)
            edges.append([id_mapping[a], id_mapping[b]])
            edges.append([id_mapping[b], id_mapping[a]])
        if not edges:
            return None
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _load_community(self, file_path, id_mapping):
        """读 `.circles`（每行一个社区，首列是社区 id），按成员数过滤后重新编号。"""
        if not os.path.isfile(file_path):
            raise Exception(f"No such file: {file_path}")
        map_comm, seen_comms = {}, set()
        with open(file_path, encoding='utf-8') as f:
            comm_id = 0
            for line in f:
                comm = line.strip().split()
                comm = comm[1:] if len(comm) > 1 else []
                if not comm:
                    continue
                comm_str = " ".join(sorted(comm))
                if comm_str in seen_comms:
                    continue
                seen_comms.add(comm_str)
                comm_nodes = [id_mapping[v] for v in set(map(int, comm)) if v in id_mapping]
                if len(comm_nodes) < self.min_community_size:
                    continue
                map_comm[comm_id] = comm_nodes
                comm_id += 1
        print(f"Loaded {len(map_comm)} communities from {file_path}")
        return map_comm

    def load_single_graph(self, dataname):
        """读一张图，返回本文件与 run.py 需要的字段（不再含 adj / 嵌入）。"""
        feat_file = os.path.join(self.root, self.dataset, f"{dataname}.feat")
        edge_file = os.path.join(self.root, self.dataset, f"{dataname}.edges")
        keep_nodes = self._nodes_in_edges(edge_file)
        if keep_nodes is not None:
            print(f'[孤立点过滤] 边文件里出现 {len(keep_nodes)} 个节点，'
                  f'仅保留这些节点（剔除无边的节点）')
        if self.attr_embed > 0:
            _cache = None
            if self.attr_cache:
                _cache = self.attr_cache_path or os.path.join(
                    self.root, self.dataset, f"{dataname}_attr_emb{self.attr_embed}.npz")
            emb, node_feats_nid, num_attrs, n_nodes, id_mapping, attr_ptr, attr_idx = \
                self._load_features_embed(feat_file, keep_nodes, self.attr_embed, _cache)
            attributes = None
        elif self.attr_topk > 0:
            attributes, node_feats_nid, node_in_dim, n_nodes, id_mapping = \
                self._load_features_topk(feat_file, keep_nodes, self.attr_topk)
            num_attrs = node_in_dim
        else:
            attributes, node_feats_nid, node_in_dim, n_nodes, id_mapping = \
                self._load_features(feat_file, keep_nodes)
            num_attrs = node_in_dim
        if n_nodes < 2:
            return None
        edge_index = self._build_edge_index(edge_file, id_mapping)
        if edge_index is None:
            return None

        cmty_file = os.path.join(self.root, self.dataset, f"{dataname}.circles")
        map_comm = self._load_community(cmty_file, id_mapping)
        if len(map_comm) < self.min_community_num:
            return None

        map_comm_attrs = {}
        for idx, community in map_comm.items():
            attr_counts = {}
            for node in community:
                for attr in node_feats_nid.get(node, []):
                    attr_counts[attr] = attr_counts.get(attr, 0) + 1
            map_comm_attrs[idx] = attr_counts

        if attributes is not None:
            node_feat, _ = build_node_feat(attributes, edge_index, n_nodes)
        else:
            node_feat = torch.cat([torch.from_numpy(emb),
                                   struct_features(edge_index, n_nodes)], dim=1)
        graph = {
            'edge_index': edge_index,
            'num_nodes': n_nodes,
            'num_attributes': num_attrs,
            'communities': map_comm,
            'community_attributes': map_comm_attrs,
            'node_feat': node_feat,
        }
        if attributes is not None:
            graph['attributes'] = attributes
        else:
            graph['attr_dim'] = int(emb.shape[1])
            graph['attr_emb'] = torch.from_numpy(emb)
            graph['attr_ptr'] = torch.from_numpy(attr_ptr)
            graph['attr_idx'] = torch.from_numpy(attr_idx)
        return graph


def build_overlap_index(map_comm):
    """节点 → 所属社区列表；社区 → 成员集合。"""
    node2comms = {}
    for cid, members in map_comm.items():
        for v in members:
            node2comms.setdefault(int(v), []).append(int(cid))
    comm_sets = {cid: set(map(int, m)) for cid, m in map_comm.items()}
    return node2comms, comm_sets


def _bfs_from(seeds, adj, kmax=4):
    """种子集到各节点的跳距（只跑到 kmax）；adj 缺失时返回空字典。"""
    if not adj:
        return {}
    dist = {int(s): 0 for s in seeds}
    frontier = set(int(s) for s in seeds)
    for k in range(1, kmax + 1):
        nxt = set()
        for u in frontier:
            for w in adj.get(u, ()):
                if w not in dist:
                    dist[w] = k
                    nxt.add(w)
        frontier = nxt
        if not frontier:
            break
    return dist


def sample_for_community(comm_id, community, node2comms, comm_sets,
                         comm_attrs_map, attr2comm, feat_idx, rng,
                         neg_rng=None, adj=None, neg_mode='rand1', neg_k=1,
                         neg_attr_k=1):
    """单个社区采样一条查询（正点/正属性规则与 sampleQuery._generate_queries 一致）。

    ---- 负信息口径（阶段32）----
    动机（实测见 temp/audit_negs_result.md）：旧口径每条查询只有 **0.85 个负点**
    （14.7% 的查询一个都没有）+ 1 个负属性，而"该被排除的歧义区域" B\\A 平均 **141 个
    节点**（p90 306）⇒ 1:166 的表达力缺口；且负点是从 B\\A 里**均匀随机**抽的（跳距分布
    与 B\\A 全体重合，不偏边界）。负信息的**位置与区分度本来就是对的**（100% 落在 B\\A、
    0 个落在目标社区内、负属性在目标社区 0 覆盖），毛病在**剂量与位置选择**上。于是把
    负点采样参数化：

      rand1  旧口径（默认）：B\\A 里均匀抽 1 个 —— 逐位复现历史数据
      randk  从 B\\A 均匀抽 k 个（只加**剂量**；作为 edgek 的对照臂）
      edgek  从"与目标社区相邻的 B\\A 节点"里抽 k 个（剂量 + **位置**）；壳层不足时按
             到种子点的跳距由近到远补齐。为什么是"相邻"：B\\A 里 62% 与目标社区有边，
             而模型实际误报落在 B\\A 的那部分就贴着目标社区。

    ⚠ **随机流纪律**（这条一破，整组数据就不能当单变量）：正点/正属性/划分必须与旧版
      逐位一致，所以 v1 那一次 `rng.choice(neg_candidates)` **照旧留在原位置、无条件执行**
      （只要候选池非空），k>1 的额外负点一律走独立的 `neg_rng`。可检查：用
      `--neg-mode rand1` 重新生成一遍，文件应与现有 .pt 完全一致
      （`temp/cmp_pt.py` 应报"负字段也全同"）。
    """
    target_set = comm_sets[comm_id]

    # ---- 正提示节点：优先重叠节点；数量不足时从社区内随机补齐 ----
    overlap_nodes = [v for v in community if len(node2comms.get(v, [])) >= 2]
    n_q = min(rng.randint(1, 3), len(community))
    if len(overlap_nodes) >= n_q:
        query_nodes = rng.sample(overlap_nodes, n_q)
    else:
        rest = [v for v in community if v not in overlap_nodes]
        query_nodes = overlap_nodes + rng.sample(rest, n_q - len(overlap_nodes))

    # ---- 负提示节点 ----
    # 候选池与旧版**逐字一致**（包括"同一节点属于多个混淆社区时在列表里出现两次"的权重
    # 效应 —— 旧口径那条单抽依赖它，重写采样时不能顺手去重）。
    confound_comms = set()
    for u in query_nodes:
        confound_comms.update(node2comms.get(u, []))
    confound_comms.discard(comm_id)
    neg_candidates = [v for cp in confound_comms for v in comm_sets[cp]
                      if v not in target_set]
    # ⚠ v1 语义的那一次抽取：位置与次数都不许变（见 docstring 的"随机流纪律"）
    _v1_neg = rng.choice(neg_candidates) if neg_candidates else None
    neg_nodes = [_v1_neg] if _v1_neg is not None else []
    if neg_mode != 'rand1' and neg_candidates:
        if neg_mode == 'edgek' and not adj:
            # 不能静默降级：没有邻接表时 edgek 会退化成"随机抽 k 个"，读数会被误读成
            # "位置无效"。宁可直接报错（main() 一定会传 adj）。
            raise ValueError('--neg-mode edgek 需要 adj（邻接表）；'
                             '请从 main() 调用或显式传入，别让它退化成 randk')
        uni = list(dict.fromkeys(neg_candidates))     # k>1 时先去重：同一节点不该进两次
        if neg_mode == 'randk':
            pool = uni
        else:                                         # edgek：优先取与目标社区相邻的壳层
            shell = [v for v in uni if adj and (adj.get(v) and (adj[v] & target_set))]
            if len(shell) < neg_k:
                dist = _bfs_from(query_nodes, adj)
                rest = sorted((v for v in uni if v not in shell),
                              key=lambda v: dist.get(v, 99))
                shell = shell + rest[:neg_k - len(shell)]
            pool = shell
        neg_nodes = (neg_rng or rng).sample(pool, min(neg_k, len(pool)))

    # ---- 正提示属性：重叠属性优先；不足由社区高频属性补齐（top-3）----
    comm_attrs = comm_attrs_map.get(comm_id, {})
    if comm_attrs:
        overlap_attrs = sorted(
            (a for a in comm_attrs
             if max((attr2comm.get(a, {}).get(c2, 0)
                     for c2 in comm_attrs_map if c2 != comm_id), default=0) > 0),
            key=lambda a: comm_attrs[a], reverse=True)
        other_attrs = sorted((a for a in comm_attrs if a not in overlap_attrs),
                             key=lambda a: comm_attrs[a], reverse=True)
        pos_attrs = (overlap_attrs + other_attrs)[:3]
    else:
        pos_attrs = []
        for v in query_nodes:
            fv = feat_idx.get(int(v))
            if fv:
                pos_attrs = fv[:3]
                break

    # ---- 负提示属性：正属性最易混淆社区里、目标社区低频的代表属性（默认 1 个）----
    # neg_attr_k = 1 时与旧版逐位相同（`cand[0]`）；> 1 时按同一排序取前 k 个不重复项。
    neg_attrs = []
    for a in pos_attrs:
        other = {c2: f for c2, f in attr2comm.get(a, {}).items() if c2 != comm_id}
        if not other:
            continue
        confound = max(other, key=other.get)
        cc = comm_attrs_map.get(confound, {})
        cand = sorted(cc, key=lambda x: (comm_attrs.get(x, 0), -cc[x]))
        if cand:
            neg_attrs = cand[:max(1, int(neg_attr_k))]
            break

    return query_nodes, pos_attrs, neg_nodes, neg_attrs


def stratified_split(order, targets):
    """把已按难度排序的社区按 3:1:1 轮转分配到 train/val/test。

    每个 5 周期贡献 3 train + 1 val + 1 test，且起点逐周期旋转，
    使三份的难度档位覆盖彼此接近（val/test 与 train 同分布）。
    order 为难度升序的社区 id 列表；返回 {cid: split}。
    """
    base = ['train', 'train', 'val', 'train', 'test']   # 3:1:1 基础模式
    rem = dict(targets)
    split_of = {}
    for i, cid in enumerate(order):
        cyc, pos = divmod(i, 5)
        for k in range(5):                 # 每个周期旋转起点，避免固定难度档位
            s = base[(pos + cyc + k) % 5]
            if rem.get(s, 0) > 0:
                split_of[cid] = s
                rem[s] -= 1
                break
        else:
            s = max(rem, key=rem.get)      # 容量耗尽时的兜底
            split_of[cid] = s
            rem[s] = max(0, rem[s] - 1)
    return split_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True,
                    help='数据集名（DataSet/<name>/，由 CommunityQueryLoader 读取）')
    ap.add_argument('--root', default='DataSet', help='数据根目录')
    ap.add_argument('--min-size', type=int, default=16, help='社区最小成员数')
    ap.add_argument('--attr-topk', type=int, default=0,
                    help='只保留文档频率最高的 K 个属性（0=全部）。twitter 这类 (n, m) 稠密特征矩阵'
                         '可达几十 GB，不裁剪时 run.py 一加载就 CUDA OOM；裁剪走两遍流式读，'
                         '节点集合、id_mapping、边/社区/查询划分都不变，只是属性维度变窄')
    ap.add_argument('--attr-embed', type=int, default=0,
                    help='用 TF-IDF + SVD 把属性压成 K 维嵌入（0=关）。写 attr_emb(n×K) + '
                         'attr_ptr/attr_idx(CSR)，不再写稠密 A；属性 id 保持原值，属性通道仍能精确算。'
                         '与 --attr-topk 互斥')
    ap.add_argument('--no-attr-cache', action='store_true',
                    help='不复用属性嵌入缓存（默认缓存在 DataSet/<name>/<name>_attr_embK.npz，'
                         '换查询参数重新生成时直接复用，不重算 SVD）')
    ap.add_argument('--attr-cache-path', default=None,
                    help='属性嵌入缓存的显式路径（默认按上面规则自动命名）')
    ap.add_argument('--per-comm', type=int, default=4,
                    help='每个社区生成的查询条数')
    ap.add_argument('--ratio', type=float, nargs=3, default=[0.6, 0.2, 0.2],
                    help='社区级 train/val/test 比例')
    ap.add_argument('--seed', type=int, default=4, help='划分与采样种子')
    ap.add_argument('--stratify', action='store_true',
                    help='分层划分：按(成员平均重叠度, 规模)排序后按 3:1:1 轮转分配，'
                         '避免难社区整批落入 val/test（让 val 与 test 难度可比）')
    ap.add_argument('--out', default=None,
                    help='输出路径（默认：<name>_csplit_queries.pt；'
                         '--neg-mode 非 rand1 时默认带模式后缀，避免覆盖基线数据）')
    ap.add_argument('--neg-mode', choices=['rand1', 'randk', 'edgek'], default='randk',
                    help='负点采样（阶段32；默认 rand1 = 旧口径，逐位复现历史数据）：'
                         'rand1=B\\A 里均匀抽 1 个；randk=从 B\\A 均匀抽 k 个（只加剂量）；'
                         'edgek=从"与目标社区相邻的 B\\A 节点"里抽 k 个（剂量+位置）。'
                         '动机：旧口径只有 0.85 个负点 vs B\\A 平均 141 个节点（1:166）。')
    ap.add_argument('--neg-k', type=int, default=3,
                    help='randk/edgek 模式每条查询的负点个数（建议 10；我们算过要覆盖模型'
                         '实际误报的那 ~26 个 B\\A 节点，k 得 ≥26）')
    ap.add_argument('--neg-attr-k', type=int, default=3,
                    help='负属性个数（默认 1 = 旧口径；规则不变：目标社区最低频优先）')
    ap.add_argument('--force', action='store_true',
                    help='允许覆盖已存在的输出文件（默认拒绝，保护既有数据）')
    ap.add_argument('--keep-adj', action='store_true',
                    help='保留图里的稠密邻接矩阵 adj（n×n float32，facebook 约 63 MB）。默认丢弃：'
                         '下游只用 edge_index 重建邻接（run.build_adj），全仓库没有读 g["adj"] 的地方')
    args = ap.parse_args()
    random.seed(args.seed)
    if args.out is None:
        out = (args.name + '_csplit_queries.pt' if args.neg_mode == 'rand1'
               else f'{args.name}_csplit_{args.neg_mode}{args.neg_k}_queries.pt')
    else:
        out = args.out
    if os.path.exists(out) and not args.force:
        raise SystemExit(f'❌ {out} 已存在；换 --out 或加 --force（默认不覆盖既有数据）')

    # ---- 用本文件自带的 CommunityQueryLoader 读取图与社区 ----
    loader = CommunityQueryLoader(args.root, args.name, args.min_size, 2,
                                  getattr(args, 'attr_topk', 0),
                                  getattr(args, 'attr_embed', 0),
                                  not getattr(args, 'no_attr_cache', False),
                                  getattr(args, 'attr_cache_path', None))
    if args.attr_topk and args.attr_embed:
        raise SystemExit('--attr-topk 与 --attr-embed 只能选一个')
    g = loader.load_single_graph(args.name)
    assert g is not None, (
        f'加载失败：检查 DataSet/{args.name}/ 下的文件与社区数（>=2）')
    n = g['num_nodes']

    map_comm = {int(cid): [int(v) for v in m]
                for cid, m in g['communities'].items()}
    comm_attrs_map = {int(cid): {int(a): int(c) for a, c in cnt.items()}
                      for cid, cnt in g['community_attributes'].items()}
    if 'attr_dim' in g:              # 压缩属性：属性 id 从 CSR 取，没有稠密 A
        _ptr = np.asarray(g['attr_ptr'], dtype=np.int64)
        _idx = np.asarray(g['attr_idx'], dtype=np.int64)
        feat_idx = {v: [int(a) for a in _idx[_ptr[v]:_ptr[v + 1]]] for v in range(n)}
    else:
        X = g['attributes']
        feat_idx = {v: [int(a) for a in np.nonzero(X[v])[0]] for v in range(n)}
    node2comms, comm_sets = build_overlap_index(map_comm)
    attr2comm = defaultdict(dict)
    for cid, cnt in comm_attrs_map.items():
        for a, c in cnt.items():
            attr2comm[a][cid] = c
    rng = random.Random(args.seed)
    # 负字段的额外抽取走**独立流**：`rng` 的消耗必须与旧版逐查询一致（见
    # sample_for_community 的"随机流纪律"），否则正点/正属性/划分会整体漂移。
    neg_rng = random.Random(args.seed * 1000 + 7)
    # 邻接表（edgek 的"是否与目标社区相邻"要用）：一次构建，全流程复用
    adj = defaultdict(set)
    _ei = g['edge_index']
    _ei = _ei.tolist() if hasattr(_ei, 'tolist') else _ei
    for _a, _b in zip(_ei[0], _ei[1]):
        adj[int(_a)].add(int(_b))
        adj[int(_b)].add(int(_a))

    # ---- 社区级三份划分 ----
    cids = sorted(map_comm.keys())
    cids = [c for c in cids if len(map_comm[c]) > 2]
    n_tr = int(len(cids) * args.ratio[0])
    n_va = int(len(cids) * args.ratio[1])
    targets = {'train': n_tr, 'val': n_va,
               'test': max(0, len(cids) - n_tr - n_va)}
    if args.stratify:
        # 难度代理（按重要性排序）：
        #   孪生强度 = 与任一其他社区的最大 Jaccard —— 决定"成员重叠"水平，
        #     是社区级划分下泛化成绩的主轴（成员近似重复的测试社区可由记忆答对）
        #   成员平均重叠度（低 = plain 查询多，基本无解）
        #   社区规模
        def hard_key(c):
            members = comm_sets[c]
            twin = max((len(comm_sets[c] & comm_sets[o])
                        / len(comm_sets[c] | comm_sets[o])
                        for o in comm_sets if o != c), default=0.0)
            ov = float(np.mean([len(node2comms[int(v)]) for v in members]))
            return (twin, ov, len(members))
        order = sorted(cids, key=hard_key)
        split_of = stratified_split(order, targets)
        cids = order
    else:
        rng.shuffle(cids)
        split_of = {}
        for i, cid in enumerate(cids):
            split_of[cid] = ('train' if i < n_tr else
                             ('val' if i < n_tr + n_va else 'test'))

    # ---- 每社区生成 --per-comm 条查询（正点/正属性规则不变；负字段按 --neg-mode）----
    queries = []
    neg_pool_sizes, neg_adj_pool, neg_counts, n_empty_pool = [], [], [], 0
    for cid in cids:
        community = map_comm[cid]
        for _ in range(args.per_comm):
            qn, pa, nn, na = sample_for_community(
                cid, community, node2comms, comm_sets,
                comm_attrs_map, attr2comm, feat_idx, rng,
                neg_rng=neg_rng, adj=adj, neg_mode=args.neg_mode,
                neg_k=args.neg_k, neg_attr_k=args.neg_attr_k)
            # 诊断字段（纯附加；run.py 不读，审计/回溯用）
            _conf = set()
            for _u in qn:
                _conf.update(node2comms.get(_u, []))
            _conf.discard(cid)
            _tgt = comm_sets[cid]
            _pool = {v for cp in _conf for v in comm_sets[cp] if v not in _tgt}
            neg_pool_sizes.append(len(_pool))
            neg_adj_pool.append(sum(1 for v in _pool if adj.get(v) and (adj[v] & _tgt)))
            neg_counts.append(len(nn))
            if not _pool:
                n_empty_pool += 1
            community_mask = torch.zeros(n)
            community_mask[torch.tensor(community, dtype=torch.long)] = 1
            queries.append({
                'query': {
                    'nodes': torch.tensor(qn, dtype=torch.long),
                    'attrs': torch.tensor(pa, dtype=torch.long),
                    'neg_nodes': torch.tensor(nn, dtype=torch.long),
                    'neg_attrs': torch.tensor(na, dtype=torch.long),
                    # 诊断字段（附加，不改语义）
                    'neg_pool': int(len(_pool)),
                    'neg_pool_adj': int(sum(1 for v in _pool
                                            if adj.get(v) and (adj[v] & _tgt))),
                    'neg_mode': args.neg_mode,
                },
                'target': {
                    'community_id': cid,
                    'community_indices': community,
                    'community_mask': community_mask,
                },
                'split': split_of[cid],
            })

    # ---- 格式 v2（阶段47）：全图信息只存顶层一份，查询里不再挂 ----
    # 旧格式把 `graph` 挂在每条查询下；pickle 虽然按对象去重，但文件里会出现 N 份引用，
    # 且任何"逐查询复制/重组"的下游都会把它变成真正的 N 份。改为顶层单份：
    #   {'graph': g, 'query_batchs': [[q, ...]], 'metadata': {...}}
    # 读取端（utils.load_queries）会把 g **共享引用**重新挂回每条查询，旧代码不用改。
    #
    # ---- 体积（阶段47b）：只写下游真正读的字段 ----
    # 以前 graph 里还带着：
    #   adj                n×n float32（facebook 3959²×4 ≈ 63 MB）—— 下游用 edge_index 重建
    #   attributes         n×num_attrs（≈22 MB），与 node_feat 的前 num_attrs 列完全重复
    #   attribute_index    node→attr 的 dict，只在**生成阶段**用
    #   community_attributes 同上
    #   node_embeddings / attr_embeddings / hyperedge_index / struct_feat  全仓库无人读取
    # 现在只写白名单；attributes 由读取端从 node_feat 切片重建（见 utils.load_queries）。
    _keep = ['num_nodes', 'num_attributes', 'edge_index', 'node_feat', 'communities']
    if 'node_feat' not in g:            # 极老的 .pt 没有 node_feat，只能保留 attributes
        _keep.append('attributes')
    if 'attr_dim' in g:                 # 压缩属性：嵌入 + CSR 索引（没有稠密 A）
        _keep += ['attr_dim', 'attr_emb', 'attr_ptr', 'attr_idx']
    if args.keep_adj:
        _keep.append('adj')
    g_save = {k: g[k] for k in _keep if k in g}
    _dropped = sorted(set(g) - set(g_save))
    print(f'  图信息：存 {sorted(g_save)}')
    print(f'          丢弃 {_dropped}'
          f'（adj 由 edge_index 重建；attributes 由 node_feat 切片重建）')
    if not args.keep_adj:
        print(f'          仅 adj 一项就省 n² float32 ≈ {n * n * 4 / 1e6:.0f} MB')
    _out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(_out_dir, exist_ok=True)     # --out 的父目录不存在时自动建
    torch.save({'graph': g_save,
                'query_batchs': [queries],
                'metadata': {'name': args.name, 'per_comm': args.per_comm,
                             'split': {s: len([q for q in queries if q['split'] == s])
                                       for s in ('train', 'val', 'test')},
                             'communities_per_split': {
                                 s: len([c for c in cids if split_of[c] == s])
                                 for s in ('train', 'val', 'test')},
                             # 负字段口径（阶段32）：回溯时不用猜是哪一版数据
                             'neg_mode': args.neg_mode,
                             'neg_k': args.neg_k,
                             'neg_attr_k': args.neg_attr_k}},
               out)

    print(f'  已保存 {out}（{os.path.getsize(out) / 1e6:.1f} MB；图只存顶层一份）')

    for s in ('train', 'val', 'test'):
        nq = len([q for q in queries if q['split'] == s])
        nc = len([c for c in cids if split_of[c] == s])
        print(f'  {s:5s}: 社区 {nc:3d}  查询 {nq:4d}（每社区 {args.per_comm} 条）')
    if args.stratify:
        print('  分层核对（三份应接近，val/test 与 train 同分布）：')
        for s in ('train', 'val', 'test'):
            cs = [c for c in cids if split_of[c] == s]
            if not cs:
                continue
            ovs = [float(np.mean([len(node2comms[int(v)]) for v in comm_sets[c]]))
                   for c in cs]
            szs = [len(comm_sets[c]) for c in cs]
            tws = [max((len(comm_sets[c] & comm_sets[o])
                        / len(comm_sets[c] | comm_sets[o])
                        for o in comm_sets if o != c), default=0.0) for c in cs]
            print(f'    {s:5s} 孪生强度 均值 {np.mean(tws):.2f}'
                  f'（≥0.3：{sum(t >= 0.3 for t in tws)}/{len(tws)}）  '
                  f'平均重叠度 {np.mean(ovs):.2f}  '
                  f'中位规模 {np.median(szs):.0f}（{min(szs)}~{max(szs)}）')
    print(f'共 {len(queries)} 条查询 → {out}')

    # ---- 负字段口径自检（阶段32）----
    # 期望：rand1 ⇒ 负点 {0:162, 1:938}、负属性全 1（与历史数据一致）；
    #       edgek10 ⇒ 负点 {0:162, 10:938}，且抽出的负点应几乎全与目标社区相邻。
    def _qs(v, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
        a = np.asarray(v, dtype=float)
        return 'mean %.1f  ' % a.mean() + '  '.join(
            'p%d %.0f' % (p * 100, np.quantile(a, p)) for p in ps)

    _cnt = defaultdict(int)
    for c in neg_counts:
        _cnt[c] += 1
    print(f'  负字段：mode={args.neg_mode} k={args.neg_k} neg_attr_k={args.neg_attr_k}')
    print(f'    负点个数分布 {dict(sorted(_cnt.items()))}'
          f'（候选池为空的查询 {n_empty_pool}/{len(queries)} = '
          f'{100.0 * n_empty_pool / max(1, len(queries)):.1f}%，这些查询必然 0 个负点）')
    print(f'    候选池大小（B\\A 去重后）{_qs(neg_pool_sizes)}；'
          f'其中与目标社区相邻的 {_qs(neg_adj_pool)}')
    if args.neg_mode == 'rand1' and args.neg_k == 1 and args.neg_attr_k == 1:
        print('    ⚠ rand1 是"逐位复现历史数据"的模式：把本文件与现有 .pt 对比，'
              '负字段也应当完全相同（temp/cmp_pt.py）')


if __name__ == '__main__':
    main()