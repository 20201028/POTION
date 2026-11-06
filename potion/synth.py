# -*- coding: utf-8 -*-
"""合成伪圈增广：在图上构造大量"按规则生成的社区"作为额外监督目标。

机制（对应三种失效假说）：
  属性聚类圈  —— 属性相似的节点组（激活属性通道，对症假说 B）
  扰动/重组圈 —— 真实圈的边界扰动与跨圈重组（对症假说 A：记忆塌缩）
  结构性圈    —— 只用图结构、**不依赖任何社区标签**生成的集合（PPR 锥 /
                 贪心凝聚扩张 / BFS 球），目标是教会"给定提示→定位局部凝聚集"
                 这一归纳偏置，对症"未见社区定位不准"（阈值标定诊断已证明
                 检索的失败不是掩码规模问题，而是概率场本身不知道社区在哪）。

监督管线：合成圈走与真实圈完全相同的查询采样（正点/负点/±属性），
仅加入训练池（val/test 永远是真实圈）。合成圈不污染 data['n2c']
（评估的重叠判定只基于真实圈）。
"""
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans

from .build_queries import sample_for_community
from .utils import conductance


def build_merged_index(real_comm, synth_members_list, feat_idx, synth_id_start):
    """训练真实圈 + 合成圈的采样索引（不污染 data['n2c']）。"""
    node2comms = defaultdict(list)
    comm_sets = {}
    comm_attrs = {}
    for cid, members in real_comm.items():
        s = set(map(int, members))
        comm_sets[int(cid)] = s
        comm_attrs[int(cid)] = Counter(a for v in s for a in feat_idx.get(int(v), []))
        for v in s:
            node2comms[int(v)].append(int(cid))
    base = synth_id_start
    assert not comm_sets or base > max(comm_sets), '合成社区 ID 与训练社区冲突'
    for i, members in enumerate(synth_members_list):
        cid = base + i
        s = set(map(int, members))
        comm_sets[cid] = s
        comm_attrs[cid] = Counter(a for v in s for a in feat_idx.get(int(v), []))
        for v in s:
            node2comms[int(v)].append(cid)
    attr2comm = defaultdict(dict)
    for cid, cnt in comm_attrs.items():
        for a, c in cnt.items():
            attr2comm[a][cid] = c
    return node2comms, comm_sets, comm_attrs, base, attr2comm


def attr_clusters(X, rng, n_out, min_size=8, max_size=400):
    """属性驱动合成圈：K-Means 聚类，簇 <min_size 丢弃、>max_size 下采样。"""
    km = KMeans(n_clusters=n_out, n_init=3, random_state=rng.randint(0, 1 << 30))
    labels = km.fit_predict(X)
    out, seen = [], set()
    for c in range(n_out):
        members = np.nonzero(labels == c)[0].tolist()
        if len(members) < min_size:
            continue
        rng.shuffle(members)
        members = members[:max_size]
        key = tuple(sorted(members))
        if key in seen:
            continue
        seen.add(key)
        out.append(members)
    return out


def perturb_real(comm_sets, adj_list, rng, n_out, min_size=8,
                 lo_j=0.15, hi_j=0.95):
    """扰动/重组圈：从真实圈出发——随机移除 10~30% 成员，再从边界邻居
    加入 10~30% 节点；保留与某真实圈 Jaccard∈[lo_j, hi_j] 的变体
    （变形但相关：太像=冗余，太不像=噪声）。
    返回 [(members, based_cid)]——based_cid 记录来源真实圈（hide 联动用）。"""
    out, seen = [], set()
    cids = list(comm_sets.keys())
    tries = 0
    while len(out) < n_out and tries < n_out * 30:
        tries += 1
        base_cid = rng.choice(cids)
        members = set(comm_sets[base_cid])
        k_drop = int(len(members) * rng.uniform(0.1, 0.3))
        for v in rng.sample(sorted(members), k=min(k_drop, len(members))):
            members.discard(v)
        if not members:
            continue
        nbrs = set()
        for v in list(members)[:300]:
            nbrs |= adj_list.get(int(v), set())
        nbrs -= members
        k_add = int(len(members) * rng.uniform(0.1, 0.3))
        if nbrs and k_add > 0:
            for v in rng.sample(sorted(nbrs), k=min(k_add, len(nbrs))):
                members.add(v)
        if len(members) < min_size:
            continue
        best = 0.0
        for c in cids:
            s2 = comm_sets[c]
            u = len(members | s2)
            if u:
                best = max(best, len(members & s2) / u)
        if not (lo_j <= best <= hi_j):
            continue
        key = tuple(sorted(members))
        if key in seen:
            continue
        seen.add(key)
        out.append((members, base_cid))
    return out


def ppr_cone(seed, src, dst, deg, n, k, alpha=0.15, iters=20):
    """个性化 PageRank 锥：以 seed 为源的 PPR 幂迭代，取概率最高的 k 个节点。

    这是"局部社区发现"的标准做法（Andersen-Chung-Lang 的近似 PPR + 扫描切分），
    完全不用任何社区标签——正是"未见社区"该有的监督形态。
    p_{t+1} = alpha*e_seed + (1-alpha)*W^T p_t，W[u,v]=1/deg(u)。
    """
    e = np.zeros(n, dtype=np.float64)
    e[seed] = 1.0
    p = e.copy()
    w = 1.0 / np.maximum(deg, 1.0)
    for _ in range(iters):
        contrib = p[src] * w[src]
        p_new = alpha * e + (1.0 - alpha) * np.bincount(dst, weights=contrib,
                                                        minlength=n)
        delta = float(np.abs(p_new - p).sum())
        p = p_new
        if delta < 1e-10:
            break
    if k >= n:
        return set(range(n))
    return set(np.argpartition(-p, k - 1)[:k].tolist())


def greedy_expand(seed, adj_list, deg, k):
    """贪心凝聚扩张：每步加入"与当前集合内部连边最多、其次度最大"的边界节点。

    保证连通，产出边界较紧的集合（与 PPR 锥互补：一个偏"随机游走可达"，
    一个偏"局部最紧"）。inner[v]=|N(v)∩S| 增量维护，避免每步重扫邻接。
    """
    S = {seed}
    inner, frontier = {}, set()
    for u in adj_list.get(seed, ()):
        inner[u] = inner.get(u, 0) + 1
        frontier.add(u)
    while len(S) < k and frontier:
        best = max(frontier, key=lambda v: (inner.get(v, 0), deg[v]))
        S.add(best)
        frontier.discard(best)
        for u in adj_list.get(best, ()):
            if u in S:
                continue
            inner[u] = inner.get(u, 0) + 1
            frontier.add(u)
    return S


def bfs_ball(seed, adj_list, k, rng):
    """BFS 球（≈ego-net）：从 seed 广度优先、同层随机顺序，取前 k 个节点。

    形态上最接近"ego 的朋友列表"，但实测**反而最不像真实圈**：随机节点的 BFS 球
    比真实朋友圈稀疏得多（conductance 中位 0.674 vs 真实社区 0.554），因为真实圈
    是"互相认识的一群人"，而 BFS 球只要求可达。故它只作为候选多样性来源，
    是否采用由 struct_circles 的 conductance 选样决定（实测被选中约 2/20）。
    """
    seen = {seed}
    order = [seed]
    i = 0
    while i < len(order) and len(order) < k:
        nbrs = list(adj_list.get(order[i], ()))
        i += 1
        rng.shuffle(nbrs)
        for u in nbrs:
            if u not in seen:
                seen.add(u)
                order.append(u)
                if len(order) >= k:
                    break
    return seen


def largest_comp_frac(S, adj_list):
    """S 内最大连通分量占 |S| 的比例（衡量生成集合是否"成团"）。"""
    if not S:
        return 0.0
    start = next(iter(S))
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for u in adj_list.get(v, ()):
            if u in S and u not in seen:
                seen.add(u)
                stack.append(u)
    return len(seen) / len(S)


def key_of(S):
    """集合的哈希键（去重用；frozenset 比 tuple(sorted(...)) 便宜）。"""
    return frozenset(S)


def conductance_ref(size_cond):
    """由 [(规模, conductance)] 构造参考函数 ref(k) = 规模相近社区的 conductance 中位。

    用途见 struct_circles：真实朋友圈是"某个 ego 的朋友列表"，比同规模的通用局部集合
    更紧（实测生成集合 conductance 中位 0.71~0.75，真实社区 0.54）；只按规模匹配不够，
    必须把 conductance 也对齐，否则合成目标的结构形态与评测目标系统性错位。
    """
    pairs = sorted(size_cond)

    def ref(k):
        near = [c for s, c in pairs if k / 2.0 <= s <= 2.0 * k]
        if not near:
            near = [c for _, c in pairs]
        return float(np.median(near)) if near else 0.5

    return ref


def struct_circles(adj_list, deg, src, dst, n, rng, n_out, sizes, min_size,
                   forbid_sets, reject_j, parts=('ppr', 'expand', 'ball'),
                   ei_np=None, ref_cond=None, cand_per_slot=6):
    """无标签结构性合成圈。返回 ([(members, kind)], stats)。

    sizes 为目标规模的经验分布（取自**训练**社区规模：分层划分下与 val/test
    同分布，且不引入评测集信息）。
    forbid_sets = val∪test 社区集合：候选若与其中任何一个 Jaccard > reject_j 则丢弃
    ——堵住"结构圈恰好等于某个评测社区"这条软泄露（KMeans 那条通道同理）。

    分布匹配：给了 ei_np + ref_cond 时，每个槽位生成 cand_per_slot 个候选，取
    |conductance − ref(规模)| 最小的那个。真实朋友圈是"某个 ego 的朋友列表"，
    比同规模的通用局部集合更紧（实测无匹配时合成集合 conductance 中位 0.63~0.68，
    真实社区 0.54），只按规模匹配不够。选样把 cond 的 IQR 从 0.37 压到 0.17、
    跨次抽样波动也从 0.54~0.70 收到 0.59~0.64；残余 +0.07 偏散是可达范围所限
    （结构生成器造不出比真实朋友圈更紧的集合），日志里会打印实测对照。

    parts 里的生成器只提供候选多样性（ppr=PPR 锥 / expand=贪心凝聚扩张 /
    ball=BFS 球）。单个生成器对 conductance 的影响**不稳定**（同生成器跨次抽样
    中位可差 0.3），所以不靠挑生成器来匹配分布，而靠上面的选样。
    """
    out, seen = [], set()
    stats = {'rej_small': 0, 'rej_forbid': 0, 'rej_dup': 0, 'gen': Counter(),
             'slots': 0}
    nodes = list(range(n))
    tries = 0
    while len(out) < n_out and tries < n_out * 40:
        tries += 1
        k = int(rng.choice(sizes))
        best, best_score, best_kind = None, None, None
        for c in range(cand_per_slot):
            kind = parts[(stats['slots'] + c) % len(parts)]
            seed = rng.choice(nodes)
            if kind == 'ppr':
                S = ppr_cone(seed, src, dst, deg, n, k)
            elif kind == 'expand':
                S = greedy_expand(seed, adj_list, deg, k)
            else:
                S = bfs_ball(seed, adj_list, k, rng)
            if len(S) < min_size:
                stats['rej_small'] += 1
                continue
            if any(len(S & f) / len(S | f) > reject_j for f in forbid_sets):
                stats['rej_forbid'] += 1
                continue
            if key_of(S) in seen:
                stats['rej_dup'] += 1
                continue
            if ei_np is None or ref_cond is None:
                best, best_kind = S, kind
                break
            mk = np.zeros(n, dtype=bool)
            mk[list(S)] = True
            score = abs(conductance(ei_np, mk, n) - ref_cond(k))
            if best_score is None or score < best_score:
                best, best_score, best_kind = S, score, kind
        stats['slots'] += 1
        if best is None:
            continue
        seen.add(key_of(best))
        out.append((sorted(best), best_kind))
        stats['gen'][best_kind] += 1
    return out, stats


def make_synth_queries(data, args, seed):
    """仅用训练社区真值生成合成查询；图特征和边仍可全图使用。"""
    rng = random.Random(seed * 7919 + 13)      # 与划分/训练种子独立偏移
    g = data['g']
    n = data['n']
    if int(getattr(args, 'synth_k', 0)) <= 0:
        return []                     # 关掉合成圈：大图上连邻接表都不建
    if 'attr_ptr' in g and 'attr_idx' in g:
        # 稠密节点表示统一存在 attr_emb 里（Cora/Twitter 是 SVD 嵌入，Amazon2M 是原始特征），
        # 离散属性 id 统一从 CSR（attr_ptr/attr_idx）取；下游与数据集无关。
        assert 'attr_emb' in g, ('该 .pt 缺少 attr_emb（稠密节点表示）；'
                                 '请用 gen_amazon2m.py 重新生成，与 Cora/Twitter 的格式对齐')
        X = np.asarray(g['attr_emb'], dtype=np.float32)   # 稠密节点表示（SVD 或原始特征）
        _ptr = np.asarray(g['attr_ptr'], dtype=np.int64)
        _idx = np.asarray(g['attr_idx'], dtype=np.int64)
        feat_idx = {v: [int(a) for a in _idx[_ptr[v]:_ptr[v + 1]]] for v in range(n)}
    else:
        X = np.asarray(g['attributes'])
        feat_idx = {v: [int(a) for a in np.nonzero(X[v])[0]] for v in range(n)}
    all_comm = {int(c): [int(v) for v in m]
                for c, m in g['communities'].items()}
    train_cids = {int(q['target']['community_id'])
                  for q in data['splits']['train']}
    assert train_cids, '合成圈增广需要至少一个训练社区'
    assert train_cids <= all_comm.keys(), '训练查询引用了图中不存在的社区'
    real_comm = {c: all_comm[c] for c in sorted(train_cids)}
    real_sets = {c: set(m) for c, m in real_comm.items()}
    adj_list = defaultdict(set)
    for a, b in zip(g['edge_index'][0].tolist(), g['edge_index'][1].tolist()):
        adj_list[a].add(b)
        adj_list[b].add(a)

    # ---- ② 属性聚类圈（激活属性通道）----
    _only = getattr(args, 'synth_only', 'both')
    if _only == 'attr':
        n_attr = args.synth_k              # 只做属性聚类
    elif _only == 'struct':
        n_attr = 0                          # 只做结构性
    else:
        n_attr = args.synth_k // 2          # 默认：各占一半
    attr_max = getattr(args, 'synth_attr_max', 400)
    attr_comms = (attr_clusters(X, rng, n_attr, min_size=8, max_size=attr_max)
                  if n_attr > 0 else [])
    # ---- ③ 扰动/重组圈（对症记忆塌缩）与 ④ 结构性圈（对症"未见社区定位"）----
    # source: perturb=旧行为（可比）；struct=只用结构生成；both=对半（默认）
    source = getattr(args, 'synth_source', 'both')
    rest = max(0, args.synth_k - len(attr_comms))
    n_pert = rest if source == 'perturb' else (rest // 2 if source == 'both' else 0)
    n_struct = rest if source == 'struct' else (rest - n_pert)
    pert_pairs = perturb_real(real_sets, adj_list, rng, n_pert, min_size=8)
    src_np = np.asarray(g['edge_index'][0].numpy(), dtype=np.int64)
    dst_np = np.asarray(g['edge_index'][1].numpy(), dtype=np.int64)
    deg_np = np.bincount(src_np, minlength=n).astype(np.float64)
    struct_pairs, sstats = [], {}
    ei_np = np.asarray(g['edge_index'].numpy())

    def _cond(S):
        mk = np.zeros(n, dtype=bool)
        mk[list(S)] = True
        return conductance(ei_np, mk, n)

    if n_struct > 0:
        # 目标规模取自训练社区（分层划分下与 val/test 同分布，且不用评测集信息）；
        # 目标 conductance 也取自训练社区（按规模分档取中位），让合成目标的结构
        # 形态与评测目标对齐——真实朋友圈比同规模的通用局部集合更紧。
        train_sizes = [len(real_sets[c]) for c in sorted(real_sets)]
        ref = conductance_ref([(len(real_sets[c]), _cond(real_sets[c]))
                               for c in sorted(real_sets)])
        forbid = [set(all_comm[c]) for c in all_comm if c not in train_cids]
        parts = tuple(p for p in
                      getattr(args, 'synth_struct_parts', 'ppr,expand,ball').split(',')
                      if p)
        struct_pairs, sstats = struct_circles(
            adj_list, deg_np, src_np, dst_np, n, rng, n_struct, train_sizes,
            getattr(args, 'synth_struct_min', 16), forbid,
            getattr(args, 'synth_reject_j', 0.5), parts, ei_np, ref,
            getattr(args, 'synth_struct_cand', 6))
    struct_comms = [m for m, _ in struct_pairs]
    # synth_entries: [(members, based_on_cids)]——属性/结构圈无来源（天然不携带
    # 任何真实社区身份，故不该被 hide 联动隐藏）；扰动圈记录来源真实圈
    synth_entries = [(m, []) for m in attr_comms] + \
                    [(m, [bc]) for m, bc in pert_pairs] + \
                    [(m, []) for m in struct_comms]
    assert all(set(based_on) <= train_cids for _, based_on in synth_entries), \
        '合成圈使用了非训练社区身份'
    synth_members = [m for m, _ in synth_entries]
    print(f'  合成圈：属性聚类 {len(attr_comms)}（≤{attr_max} 节点）+ '
          f'扰动重组 {len(pert_pairs)}（扰动来源仅限 {len(train_cids)} 个训练社区）'
          f' + 结构性 {len(struct_comms)}（--synth-source {source}）')
    if struct_pairs:
        ss = [len(S) for S, _ in struct_pairs]
        sf = [largest_comp_frac(set(S), adj_list) for S, _ in struct_pairs]
        tr_s = [len(real_sets[c]) for c in real_sets]
        tr_c = [_cond(real_sets[c]) for c in real_sets]
        tr_f = [largest_comp_frac(real_sets[c], adj_list) for c in real_sets]
        print(f'    构成 {dict(sstats["gen"])}；候选拒绝：过小 {sstats["rej_small"]} / '
              f'撞 val-test {sstats["rej_forbid"]}（Jaccard>'
              f'{getattr(args, "synth_reject_j", 0.5)}）/ 重复 {sstats["rej_dup"]}')
        print(f'    规模 min/中位/max {min(ss)}/{int(np.median(ss))}/{max(ss)}'
              f'  vs 训练社区 {min(tr_s)}/{int(np.median(tr_s))}/{max(tr_s)}')
        print(f'    conductance 中位 {np.median([_cond(set(S)) for S, _ in struct_pairs]):.3f}'
              f'（各生成器 '
              f'{" ".join("%s %.3f" % (k, np.median([_cond(set(S)) for S, kk in struct_pairs if kk == k])) for k in sorted(sstats["gen"]))}）'
              f' vs 训练社区 {np.median(tr_c):.3f}')
        print(f'    最大连通分量占比 中位 {np.median(sf):.2f} vs 训练社区 '
              f'{np.median(tr_f):.2f}')

    # ---- 合并索引（采样用；不影响评估的 n2c）----
    node2comms, comm_sets, comm_attrs, base, attr2comm = build_merged_index(
        real_comm, synth_members, feat_idx, max(all_comm) + 1)
    assert all_comm.keys().isdisjoint(range(base, base + len(synth_entries))), \
        '合成社区 ID 与真实社区冲突'

    # ---- 每合成圈 2 条查询（与真实查询同构）----
    queries = []
    for i, (members, based_on) in enumerate(synth_entries):
        cid = base + i
        community = sorted(members)
        community_mask = torch.zeros(n)
        community_mask[torch.tensor(community, dtype=torch.long)] = 1
        for _ in range(2):
            qn, pa, nn, na = sample_for_community(
                cid, community, node2comms, comm_sets,
                comm_attrs, attr2comm, feat_idx, rng)
            q = {
                'graph': g,
                'query': {
                    'nodes': torch.tensor(qn, dtype=torch.long),
                    'attrs': torch.tensor(pa, dtype=torch.long),
                    'neg_nodes': torch.tensor(nn, dtype=torch.long),
                    'neg_attrs': torch.tensor(na, dtype=torch.long),
                },
                'target': {
                    'community_id': cid,
                    'community_indices': community,
                    'community_mask': community_mask,
                },
                'based_on': based_on,          # 来源真实圈（hide 联动用）
                'synth': True,
            }
            q['_full'] = community_mask.bool()
            queries.append(q)
    return queries
