# -*- coding: utf-8 -*-
"""直接读取 build_queries.py 生成的 _queries.pt：训练 + 评估 + 探测一体运行。

默认不产生任何模型中间文件（最优参数在内存中回载）；
如需保存供复用，加 --save（写入 --out 目录的 best.pt / config.json）。

流程：读查询 → 两轮交互训练（第 1 轮提示形态采样=主监督，错误驱动补点，
第 2 轮辅助监督）→ 早停取最优参数 → 检索精度(±负提示 × 重叠分组) /
负提示消歧增益 / 模拟交互增益曲线 / ACG 探测 / 时延微基准。

用法：python -m potion.train_eval --queries data/cora_queries.pt
"""
# ---- 诊断输出过滤（只保留主指标与 CLARE 口径的 ACG 行）----
_HIDDEN_DIAG = ('稳定性分数示例', 'plain 组按社区拆开', '基础能力诊断', '塌缩诊断',
                '假阳性质量分解', '负提示消歧增益', '按成员重叠分组', '按种子覆盖分组',
                '节点级 micro-F1', '形状诊断', '逐查询尺度选择诊断', '阈值标定诊断',
                '阈值扫描')
_orig_print = print
_diag_state = {'hide': False, 'acg': False}


def _filtered_print(*a, **k):
    s = ' '.join(str(x) for x in a)
    t = s.strip()
    if t.startswith('== '):
        _diag_state['hide'] = any(h in t for h in _HIDDEN_DIAG)
        _diag_state['acg'] = ('ACG 探测' in t)
        if _diag_state['hide']:
            return
    if _diag_state['hide']:
        return
    if (_diag_state['acg'] and t and not t.startswith('== ')
            and 'CLARE' not in t and '[time]' not in t):
        return
    _orig_print(*a, **k)


print = _filtered_print


import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from .model import CAM
from .utils import (set_seed, dice_loss, tversky_loss, ppr_score, prf, load_queries,
                   make_prompt, make_prompt_noneg, make_prompt_attr,
                   make_prompt_cat, make_prompt_rand, has_pos_node,
                   PROMPT_CATS, NEG_CATS, POSONLY_CATS, cat_weights,
                   error_driven_prompts,
                   split_queries, degree_features, conductance, stability_score,
                   iou)


# ==================== 数据 ====================
def build_node2comms(communities):
    n2c = defaultdict(list)
    for cid, members in communities.items():
        for v in members:
            n2c[int(v)].append(int(cid))
    return n2c


def hop_ball_mask(seeds, adj, n, k):
    """以 `seeds` 为起点的 k 跳邻域（bool 掩码，长度 n）；k<=0 或无种子时返回 None。

    ⚠ adj 是 `build_adj` 返回的 **list of list**（不是 dict），所以取邻居用 `adj[u]`，
    不能写 `adj.get(...)`（列表没有 .get）。

    用途（`--restrict-hop`）：这是一条**输出规则**，把预测掩码限制在种子点的局部邻域内，
    使"输出一个远处的记忆社区"在结构上不可能。动机来自阶段12 的基础能力诊断：
    **plain 组有 80.2% 的掩码落在种子点 2 跳之外**（全体均值 41.8% / 中位 21.8%），
    而 plain 的答案就在种子身边——那 80% 正是"检索了记忆库"的结构指纹。
    无标签：只用图结构与提示本身。
    """
    if k <= 0 or not seeds:
        return None
    ball = set(int(v) for v in seeds)
    frontier = set(ball)
    for _ in range(k):
        nxt = set()
        for u in frontier:
            nxt.update(adj[u])
        nxt -= ball
        if not nxt:
            break
        ball |= nxt
        frontier = nxt
    m = np.zeros(n, dtype=bool)
    m[list(ball)] = True
    return m


def prompt_ball(plist, data, args):
    """提示的**正点**张成的 k 跳球（带缓存）。`--restrict-hop 0` 或无正点时返回 None。

    只取正点（kind==0 & pol==0）：负提示点、属性提示都不该扩张邻域。
    缓存键是正点 id 的排序元组 —— 检索侧同一提示会在 8 个诊断块里被反复用到，
    不缓存就白做 8 倍的 BFS；ACG 侧两个模式（node-only / n+a）也会重复同一批种子。
    """
    if args.restrict_hop <= 0 or plist is None:
        return None
    nodes = tuple(sorted({int(p['id']) for p in plist
                          if p['kind'] == 0 and p['pol'] == 0}))
    if not nodes:
        return None
    # ⚠ 缓存键必须**带上 k**：只按正点元组做键的话，同一进程里换了 k 会把旧的球拿回来。
    #   当前 k 是 CLI 参数、整轮不变所以不会触发，但"靠调用方保证"是隐性契约，
    #   验证脚本一次就把它逼出来了（k=1 拿到 k=2 的球）。键里加 k 就没有这个隐患。
    key = (nodes, args.restrict_hop)
    # ⚠ 变量名刻意不用裸 `cache`：check_py38.py 按词边界找 py3.9+ 名字（`functools.cache`），
    #   一个叫 cache 的局部变量会被误报。用 bcache 与 prior_cache 的命名一致。
    bcache = data.setdefault('ball_cache', {})
    if key not in bcache:
        bcache[key] = hop_ball_mask(nodes, data['adj'], data['n'], args.restrict_hop)
    return bcache[key]


def build_alt_targets(q, n2c, comm_mask, allowed=None):
    """查询的**候选目标**列表（方案 4，K>1 粒度用）：`[A] ∪ 种子点的其它社区`。

    为什么这样定义：种子点被 build_queries 刻意选成"属于 ≥2 个社区"的重叠点，
    所以提示本身有歧义 —— 单掩码的最优解必然是几个候选的折中集（RUN-LIST Q4(b)）。
    SAM 对提示歧义的原生答案就是给多个掩码，这里让 K 个粒度 token 分别承诺一个候选。

    ⚠⚠ `allowed` 是**必须的防泄漏闸门**，别省略：候选的混淆社区若取自全部 55 个社区，
    那么某条**训练**查询的种子点只要同时属于一个 test 社区，那个 test 社区的**完整掩码**
    就成为训练目标 —— 训练用到了测试标签，测试集不再是"未见社区"，检测侧 found/AvgF1 会被虚高。
    改动前（目标恒为 A）不会有这个问题，因为训练查询的 A 一定是 train 社区。
    所以调用方必须传 `allowed = train 划分的社区集合`（见 load_raw）。
    `allowed=None` 只在诊断/复现时用（表示不过滤），正常训练路径**不允许**为 None。

    返回 list[Tensor(bool)]，第 0 个恒为 A（查询真值）。顺序是**规范序**（A 在前、其余按 id 升序）；
    训练时的槽位顺序会另行打乱，见 train()。
    """
    cid = int(q['target']['community_id'])
    seeds = [int(v) for v in q['query']['nodes'].tolist()]
    others = sorted({int(c) for v in seeds for c in n2c.get(v, [])} - {cid})
    out = [q['_full']]
    for c in others:
        if c in comm_mask and (allowed is None or c in allowed):
            out.append(comm_mask[c])
    return out


def load_raw(path, device):
    """读取查询文件并构建与 seed 无关的数据（feat/edge_index/覆盖掩码）。

    节点特征优先用 .pt 里预先算好的 `node_feat`（由 build_queries.py /
    sampleQuery.py 在生成查询时写入：原属性 ⊕ 结构特征）。特征属于数据，
    训练脚本不再自己拼——否则同一个 .pt 被不同脚本读到会得到不同的输入。
    旧 .pt 没有该字段时退回"属性 ⊕ log-degree one-hot"，并打印提示。
    """
    qs, g = load_queries(path)
    n = int(g['num_nodes'])
    num_attrs = int(g['num_attributes'])
    attr_dim = int(g['attr_dim']) if 'attr_dim' in g else num_attrs
    edge_index = g['edge_index'].long().to(device)
    if 'node_feat' in g:
        feat = torch.as_tensor(np.asarray(g['node_feat']),
                               dtype=torch.float32).to(device)
        if 'attr_emb' in g:
            _head = f'{attr_dim} 维属性（TF-IDF+SVD 压缩自 {num_attrs} 维）'
        elif 'attr_ptr' in g:
            _head = f'{attr_dim} 维输入特征（{num_attrs} 维离散属性另行索引）'
        else:
            _head = f'{attr_dim} 维属性'
        print(f'== 节点特征 ==  取自 .pt 的 node_feat：'
              f'{_head} ⊕ {feat.size(1) - attr_dim} 维结构'
              f'（共 {feat.size(1)} 维）')
    else:
        x = torch.as_tensor(np.asarray(g['attributes']), dtype=torch.float32)
        feat = torch.cat([x, degree_features(g['edge_index'], n)], dim=1).to(device)
        print(f'== 节点特征 ==  ⚠ .pt 里没有 node_feat 字段，退回旧口径'
              f'（属性 {num_attrs} 维 ⊕ log-degree one-hot 16 维 = {feat.size(1)} 维）；'
              f'请用 potion.build_queries 重新生成查询文件以统一特征')
    for q in qs:
        q['_full'] = q['target']['community_mask'].bool()
    # 方案 4：预建候选目标（每条查询一次，训练/评测共用同一份）
    # ⚠ 候选掩码保持 **CPU**：`q['_full']` 本来就是 CPU（compute_loss 里再 .to(device)），
    # 两者必须同设备，否则 _alts 里会混着 CPU/GPU 张量，后面 `.to(device)` 行为不一致。
    n2c = build_node2comms(g['communities'])
    comm_mask = {}
    for c, members in g['communities'].items():
        mk = torch.zeros(n, dtype=torch.bool)
        mk[torch.as_tensor(list(members), dtype=torch.long)] = True
        comm_mask[int(c)] = mk
    # ---- 防泄漏：候选里的混淆社区**只允许 train 划分** ----
    # 详见 build_alt_targets 的 docstring。这里顺便把"不过滤时会漏多少"统计出来打印，
    # 因为那个数字就是改动前的泄漏面大小（也解释了 K=3 首跑 found 为什么突然到 6/11）。
    split_of = {int(q['target']['community_id']): q.get('split') for q in qs}
    if any(s is None for s in split_of.values()):
        # 不能静默降级成"不过滤" —— 那等于悄悄关掉防泄漏闸门。
        raise ValueError(
            '.pt 里缺少社区级 split 字段，无法确定 train 白名单，防泄漏闸门无法生效。'
            '请用 potion.build_queries 重新生成查询文件（带 split），'
            '或在不需要 K>1 时用 --k-gran 1。')
    train_cids = {c for c, s in split_of.items() if s == 'train'}
    q_train = [q for q in qs if q['split'] == 'train']
    leak_q, leak_test_cids = 0, set()
    for q in q_train:
        cid = int(q['target']['community_id'])
        seeds = [int(v) for v in q['query']['nodes'].tolist()]
        others = {int(c) for v in seeds for c in n2c.get(v, [])} - {cid}
        bad = {c for c in others if split_of.get(c) in ('val', 'test')}
        if bad:
            leak_q += 1
        leak_test_cids |= {c for c in bad if split_of.get(c) == 'test'}
    for q in qs:
        q['_alts'] = build_alt_targets(q, n2c, comm_mask, allowed=train_cids)
    n_multi = sum(1 for q in qs if len(q['_alts']) > 1)
    n_alt = np.mean([len(q['_alts']) for q in qs])
    n_test = sum(1 for s in split_of.values() if s == 'test')
    print(f'== 候选目标（K 粒度）==  查询 {len(qs)}；有 ≥2 个候选的 {n_multi} 条'
          f'（{n_multi / max(1, len(qs)) * 100:.1f}%）；平均候选数 {n_alt:.2f}')
    print(f'  [防泄漏] 混淆社区仅限 train 划分（{len(train_cids)} 个社区）')
    print(f'  ⚠ 若**不**加这道闸门，训练池 {len(q_train)} 条里有 {leak_q} 条会把 val/test 社区'
          f'当候选（其中 test 社区 {len(leak_test_cids)}/{n_test} 个）'
          f'—— 那等于把测试社区的掩码当训练标签，found 会被虚高')
    ei_np = g['edge_index'].numpy()
    # ---- 压缩属性（--attr-embed）的 CSR + 反向索引 ----
    # 属性通道与 ACG 只需要"节点→属性 id"，不需要稠密 A：
    #   attr_ptr/attr_idx  节点 → 属性列表（CSR）
    #   attr_rev_ptr/attr_rev_nodes  属性 → 节点列表（反向索引，属性场按它 O(nnz) 累加）
    if 'attr_dim' in g:
        _ptr = np.asarray(g['attr_ptr'], dtype=np.int64)
        _idx = np.asarray(g['attr_idx'], dtype=np.int64)
        _cnt = np.diff(_ptr)
        _node_of = np.repeat(np.arange(n, dtype=np.int64), _cnt)
        _order = np.argsort(_idx, kind='stable')
        _rev_nodes = _node_of[_order]
        _rev_ptr = np.concatenate(
            [[0], np.cumsum(np.bincount(_idx[_order], minlength=num_attrs))]).astype(np.int64)
    else:
        _ptr = _idx = _cnt = _rev_ptr = _rev_nodes = None
    return dict(qs=qs, g=g, n=n, num_attrs=num_attrs, attr_dim=attr_dim,
                attr_ptr=_ptr, attr_idx=_idx, attr_cnt=_cnt,
                attr_rev_ptr=_rev_ptr, attr_rev_nodes=_rev_nodes,
                edge_index=edge_index,
                feat=feat, n2c=n2c, comm_mask=comm_mask,
                # 候选目标的防泄漏白名单（train 划分）；合成圈也要用它，
                # 所以存进 data 供 main 里生成 synth_qs 后复用
                alt_allowed=train_cids,
                ei_np=ei_np,
                # 先验用（ppr_score 只按 src 取度，与 utils.struct_features 同约定）
                ei_src=ei_np[0], ei_dst=ei_np[1],
                deg=np.bincount(ei_np[0], minlength=n).astype(np.float64),
                prior_cache={},
                ball_cache={},
                adj=build_adj(g['edge_index'], n))


# ==================== 查询局部先验（方案 2+3）====================
def query_prior(plists, data, args):
    """提示 -> (B,2,n) 逐节点先验：[:,0]=正点 PPR 锥、[:,1]=负点 PPR 锥（各按最大值归一化）。

    为什么要有它（见 RUN-LIST 阶段8 的诊断）：种子点被刻意选成"属于 ≥2 个社区"的重叠点，
    PPR 从它出发会同时铺满目标 A 与混淆社区 B，且**质心系统性偏向 B**（同号比例仅 32.5%）。
    而负提示恰好给了一个落在 B\\A 里的节点，所以 `正锥 − λ·负锥` 把 A 的 Dice 从 0.427
    抬到 0.461（λ≈0.5~1，倒 U 型）。两个信任度由 `model.decoder.prior_w` 学习，初值 0。

    ⚠ 训练/测试同构的关键：必须用**实际送进解码器的那份提示**。训练提示被
    `sample_train_prompt` 抽稀过（full/positive/seed/subset 四种形态），若用
    `q['query']['nodes']` 就是"训练看全种子的锥、测试看测试提示的锥"，属训练/测试不一致。

    先验本身**不含任何标签**（只用图结构 + 提示），故训练/测试可同构计算。
    """
    if args.prior == 'none' or not plists or data is None or args is None:
        return None
    pcache = data['prior_cache']
    n = data['n']
    src, dst, deg = data['ei_src'], data['ei_dst'], data['deg']
    dev = data['feat'].device
    rows = []
    for pl in plists:
        pos = tuple(sorted({int(p['id']) for p in pl
                            if p['kind'] == 0 and p['pol'] == 0}))
        neg = tuple(sorted({int(p['id']) for p in pl
                            if p['kind'] == 0 and p['pol'] == 1}))
        if args.prior_neg != 'cone':
            neg = ()
        key = (pos, neg)
        w = pcache.get(key)
        if w is None:
            wp = ppr_score(pos, src, dst, deg, n,
                           args.prior_alpha, args.prior_iters) if pos else np.zeros(n)
            mx = wp.max()
            wp = wp / mx if mx > 0 else wp
            wn = np.zeros(n)
            if neg:
                wn = ppr_score(neg, src, dst, deg, n,
                               args.prior_alpha, args.prior_iters)
                mx = wn.max()
                wn = wn / mx if mx > 0 else wn
            # 负锥取负号存进第 1 通道（模型侧是 γ·w0 − λ·w1，λ 可学、初值 0）。
            # ⚠ **缓存一律放 CPU（主机内存）**：首跑实测缓存会长到 9.8 万条（≈3.1 GB）——
            # 我原先估的"约 250MB、有界"错了 12 倍，原因是 (a) sample_train_prompt 每轮
            # 重抽子集、(b) error_driven_prompts 每轮往提示里补点并带入下一轮，
            # 两者叠加让键空间爆炸；(c) 缓存挂在 data 上、跨 seed 累积。
            # 放 CPU 后同样的 3.1GB 落在主机内存（便宜且不会 OOM 显存），
            # 每次调用一次性搬到设备（batch 16 约 0.5 MB，开销可忽略）。
            # 注意：缓存只是 memoization，值的计算与缓存位置无关 → 不改变任何结果。
            w = torch.as_tensor(np.stack([wp, -wn], axis=0), dtype=torch.float32)
            pcache[key] = w
        rows.append(w)
    return torch.stack(rows).to(dev)          # (B,2,n)


# ==================== 相对位置通道（阶段12）====================
POS_DIM = {'hop': 5, 'ppr': 1, 'both': 6}


def pos_dim_of(chan):
    """位置通道名 -> 维度：hop=5（0/1/2/3/≥4 五桶 one-hot）、ppr=1（归一化 log-PPR）、both=6。"""
    if chan not in POS_DIM:
        raise ValueError('未知的位置通道 %r' % (chan,))
    return POS_DIM[chan]


SEED_SIM_DIM = {'off': 0, 'mean': 1, 'max': 1, 'both': 2}


def seed_sim_dim(args):
    """种子内容相似度通道的维度（阶段37）：off=0 / mean=1 / max=1 / both=2。"""
    mode = getattr(args, 'seed_sim', 'off')
    if mode not in SEED_SIM_DIM:
        raise ValueError('未知的 --seed-sim 取值 %r' % (mode,))
    return SEED_SIM_DIM[mode]


def geo_dim_total(args):
    """几何块本身的维度 = 正参照 +（可选）负参照（不含种子相似度块）。"""
    d = pos_dim_of(args.pos_chan)
    if getattr(args, 'pos_neg', 'off') != 'off':
        d += pos_dim_of(args.pos_neg)
    return d


def pos_dim_total(args):
    """位置通道的总维度 = 几何块 +（可选的）种子相似度块。

    负参照是**另一块拼在后面**（不是同一块加权求和）：给模型两块独立几何场，让它自己学
    "正的在哪儿"和"负的在哪儿"怎么组合 —— 这正是方案 3 的加性标量场 γ·w_pos − λ·w_neg
    做不到的（那是先替模型定死了组合方式）。

    ⚠ 阶段37 起这里**还包含** --seed-sim 的块（默认 off ⇒ +0），而 `pos_feat()` 只返回几何块、
    相似度块由 `query_pos()` 拼在后面 —— 两处的宽度之和必须等于本函数（守卫在
    temp/verify_seed_sim.py）。
    """
    return geo_dim_total(args) + seed_sim_dim(args)


def seed_sim_block(pos, H, args):
    """提示正点 -> (n, seed_sim_dim) 的**内容相似度**场：cos(H_v, mean_{正点} H) 与/或
    max_{正点} cos(H_v, H_s)，H = 编码器输出。

    为什么有它（阶段37 的离线体检，220 条 test 查询；脚本 temp/prompt_src_diag.py）：
      · **单独用**相似度场 = "把节点 token 换成 H[v]" 的近似效果 ⇒ P@|A| 0.387 vs hop 的 0.353，
        但配对胜负 106胜/91负（≈平手）⇒ 不够；
      · **叠加**（hop + 0.25·相似度）⇒ P@|A| **0.400**、配对 **127胜/58负**；而且受益在
        `unseen 种子`（+0.046）与 `成员重叠<0.3`（+0.040）两组**同样出现** ⇒ 不是记忆捷径；
        随机种子/随机属性的对照场只有 0.018~0.020（零假设干净）。
      · 两种场是**互补**的：相似度场的 AUC 更低（0.887 < 0.960）却 top-|A| 命中更高 ——
        所以是"叠加"，不是"替换"。
      · 属性侧的对应物（把属性 token 换成其携带者的 H 池化）实测死透（P@|A| 0.017~0.022），
        不做。

    ⚠ H **detach**：a) 复现离线测量（那次用的是冻结的编码器输出）；b) 不给编码器开第二条
      梯度路径 —— 相似度若带梯度，会同时拉扯编码器（本通道只该作为"输入特征"，学习的是
      注入它的那层投影，与几何块同一纪律）。
    ⚠ **不进 pos_cache**：它只依赖 (pos, H)，每次 O(n·d) 且 H 每图只算一次 ⇒ 便宜；更重要的是
      避免"缓存键漏带 seed_sim 配置"这类坑（本文件已为同类问题写过两次注释）。
    ⚠ 与 `--pos-mode off` 不兼容（同一个注入点）—— main 里有 ap.error 拦。
    """
    mode = getattr(args, 'seed_sim', 'off')
    if mode == 'off':
        return None
    if H is None:
        raise ValueError('--seed-sim 需要 H（编码器输出）；请从 decode() 调用并传入 H')
    if H.dim() != 2:
        raise ValueError('seed_sim_block 期望 H 是 (n,d)，收到 %r' % (tuple(H.shape),))
    with torch.no_grad():
        Hn = torch.nn.functional.normalize(H.detach(), dim=1, eps=1e-9)
        idx = torch.as_tensor(sorted(int(v) for v in pos), dtype=torch.long,
                              device=Hn.device)
        cols = []
        if idx.numel() == 0:
            # 没有正点（不该出现：正节点必备）⇒ 该块给 0，不静默丢维度
            return torch.zeros(Hn.size(0), seed_sim_dim(args), device=Hn.device)
        if mode in ('mean', 'both'):
            m = Hn[idx].mean(dim=0)
            m = m / (m.norm() + 1e-9)
            cols.append((Hn @ m).unsqueeze(1))
        if mode in ('max', 'both'):
            cols.append((Hn[idx] @ Hn.t()).max(dim=0).values.unsqueeze(1))
        return torch.cat(cols, dim=1)


class LazyHopTable:
    """按需计算 hop 行（只算被查询用到的种子）。

    mode='khop'：dijkstra 带 limit=buckets-1，只展开 <=k 跳（默认，快）；
    mode='full'：不设 limit，算精确最短路，再截断到 buckets-1（慢，语义与 khop 相同，
                 但 buckets 很大时才有区别）。
    两种模式都按需计算、按行缓存。
    """

    def __init__(self, edge_index, n, buckets=5, cap_cache=8000, mode='khop'):
        import scipy.sparse as sp
        from scipy.sparse.csgraph import dijkstra
        src = np.asarray(edge_index[0], dtype=np.int64)
        dst = np.asarray(edge_index[1], dtype=np.int64)
        self.A = sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)),
                               shape=(n, n))
        self._dij = dijkstra
        self.n, self.cap, self.cap_cache = n, buckets - 1, cap_cache
        self.mode = mode
        self.cache = {}

    def _rows(self, seeds):
        kw = dict(directed=False, indices=seeds, unweighted=True)
        if self.mode == 'khop':
            kw['limit'] = self.cap
        Dm = self._dij(self.A, **kw)
        Dm = np.nan_to_num(Dm, nan=self.cap, posinf=self.cap)
        return np.minimum(Dm, self.cap).astype(np.uint8)

    def __getitem__(self, idx):
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        need = [int(s) for s in idx if int(s) not in self.cache]
        if need:
            Dm = self._rows(need)
            for i, s in enumerate(need):
                if len(self.cache) < self.cap_cache:
                    self.cache[s] = Dm[i]
        rows = []
        for s in idx:
            s = int(s)
            if s in self.cache:
                rows.append(self.cache[s])
            else:                                   # 超出缓存上限：现算不缓存
                rows.append(self._rows([s])[0])
        return np.stack(rows)


def hop_dist_table(adj, n, buckets=5):
    """全源 BFS 距离表 (n,n) uint8，距离按 buckets-1 截断（末桶 = "≥buckets-1 或不可达"）。

    为什么预计算整张表、而不是逐提示算 + 缓存：hop 通道的键空间是"提示正点集合"，而训练
    提示每轮重抽子集（p_drop=0.7）→ 键空间会像先验那样爆掉（先验缓存实测到过 9.8 万条 /
    3.1 GB）。距离矩阵只依赖图、与提示无关：一次算完（实测 n=3959、平均度 20 时 **10 秒**、
    15.7 MB；稀疏图更快），之后每条查询只是一次行内取最小（多正点取 min ≡ 多源 BFS），
    **无缓存、无键空间**。见 temp/time_hop_table.py（同规模随机图上的计时）。

    ⚠ 代价是 O(n²) 内存：n 上万时这张表到几百 MB，那时请改用 --pos-chan ppr。
    """
    cap = buckets - 1
    D = np.full((n, n), cap, dtype=np.uint8)
    for s in range(n):
        # ⚠ 内层循环刻意用 bytearray + Python list，最后一次性写回 D[s]：
        #   `seen` 用 numpy 数组、或在内层做 `D[s, w] = nd` 这种**逐元素 numpy 标量赋值**，
        #   每次下标都要走一遍 numpy 的标量协议（~200ns），n=4 千时整个表要多花几十秒。
        #   这里把 O(n·E) 次内层迭代全放在纯 Python 里，每个源只做一次花式索引赋值。
        seen = bytearray(n)
        seen[s] = 1
        nodes, dists = [s], [0]
        frontier = [s]
        dist = 0
        while frontier and dist + 1 < buckets:
            nd = dist + 1
            nxt = []
            for u in frontier:
                for w in adj[u]:
                    if not seen[w]:
                        seen[w] = 1
                        nodes.append(w)
                        dists.append(nd)
                        nxt.append(w)
            frontier = nxt
            dist = nd
        D[s, np.asarray(nodes, dtype=np.int64)] = np.asarray(dists, dtype=np.uint8)
    return D


def rel_block(ref, data, chan, args, buckets=5, nodes=None):
    """一个参照点集合 `ref`（节点 id）在给定通道下的相对位置块 (n, d)；ref 为空 → 全 0。

    hop：到最近参照点的跳距 one-hot（用预计算的全源距离表；多源 = 各行取 min ≡ 多源 BFS）。
    ppr：从参照点出发的 PPR，先按最大值归一、再取 log/10（重尾被压到 ≈[-1.85, 0]）。
    两个通道都只依赖图 + 提示，**不含标签**，故训练/测试同构。

    ⚠ **无参照点 → 整块留 0**：这是"**没有**位置信息"，而**不是**"所有节点都最远"。
    后者是一个（错误的）位置断言，会让模型在没有参照点时也收到确定的信号。
    """
    n = data['n']
    parts = []
    if chan in ('hop', 'both'):
        nn = n if nodes is None else int(len(nodes))
        onehot = np.zeros((nn, buckets), dtype=np.float32)
        if ref and nodes is not None:
            # 候选集内多源 BFS：只算种子到 C 内节点的跳距，不建全源表
            sel = np.asarray(nodes, dtype=np.int64)
            pos_of = np.full(n, -1, dtype=np.int64)
            pos_of[sel] = np.arange(sel.size)
            cap = buckets - 1
            dd = np.full(sel.size, cap, dtype=np.int64)
            frontier, seen = [], set()
            for v in ref:
                v = int(v)
                if pos_of[v] >= 0 and v not in seen:
                    seen.add(v); dd[pos_of[v]] = 0; frontier.append(v)
            for d in range(1, cap + 1):
                nxt = []
                for u in frontier:
                    for w in data['adj'][u]:
                        w = int(w)
                        if w not in seen and pos_of[w] >= 0:
                            seen.add(w); dd[pos_of[w]] = d; nxt.append(w)
                frontier = nxt
                if not frontier:
                    break
            onehot[np.arange(nn), dd] = 1.0
        elif ref:
            D = data.get('hop_dist')
            if D is None:
                D = data['hop_dist'] = LazyHopTable(
                data['edge_index'].cpu().numpy(), n, buckets,
                mode=getattr(args, 'hop_mode', 'khop'))
            dd = D[np.asarray(ref, dtype=np.int64)].min(axis=0)
            onehot[np.arange(nn), dd] = 1.0
        parts.append(onehot)
    if chan in ('ppr', 'both'):
        if ref:
            pv = ppr_score(ref, data['ei_src'], data['ei_dst'], data['deg'], n,
                           args.prior_alpha, args.prior_iters)
            mx = pv.max()
            pv = pv / mx if mx > 0 else pv
            pv = np.log(np.maximum(pv, 1e-8)) / 10.0
        else:
            pv = np.zeros(n)
        parts.append(pv[:, None].astype(np.float32))
    return np.concatenate(parts, axis=1)


def pos_feat(pos, neg, data, args, nodes, buckets=5):
    """提示正点/负点 -> (n, pos_dim_total) 的相对位置特征：前半 = 从正点出发，后半 = 从负点出发。

    负参照（`--pos-neg`）的动机（阶段14）：假阳性质量分解显示 **51.4% 的 FP 质量**是种子点的
    **其它社区 B\\A**，而负提示点恰好就落在那里；可负提示目前几乎没用（负提示增益 all 仅
    +0.001）—— 信息在提示里，模型拿不到。方案 3 曾把同一个量做成**加性标量场** γ·w_pos − λ·w_neg
    而失败；本通道把它做成解码器的**输入特征**（与正通道完全同一套定义、同一处注入），
    也就是"方案3 失败在注入方式、不在那个量"这句话的直接检验。
    """
    parts = [rel_block(pos, data, args.pos_chan, args, buckets, nodes)]
    if getattr(args, 'pos_neg', 'off') != 'off':
        parts.append(rel_block(neg, data, args.pos_neg, args, buckets, nodes))
    return np.concatenate(parts, axis=1)


def query_pos(plists, data, args, H=None, nodes=None):
    """提示 -> (B,n,pos_dim) 逐节点**相对位置** r(v | 提示的正点)；--pos-mode off 时返回 None。

    为什么必须有它（阶段12 的诊断）：掩码 = ⟨h_v, e⟩/√d 里没有任何一项依赖 (节点, 提示) 的
    **关系**，所以模型只能检索训练时记住的社区 —— 98% 查询的最佳匹配不是目标；而把输出限制
    到种子 2 跳球后，plain 20 条里 15 条直接变空（**球内概率≈0**）。位置因此必须进"算掩码的
    那一步"，而且**训练时就得在**（这点与 --restrict-hop 相反：那条只在评测生效，训练从没
    见过位置，所以学不出任何东西，只能把掩码裁空）。

    为什么是**相对**位置：单张固定图上，逐节点的绝对位置表 ≈ 节点身份 ≈ 记忆句柄（SAM 的
    image_pe 安全，只因为它那个网格是**所有图片共享**的）。r(v|seed) 平移不变、不回答
    "我是谁"；而"答案就是离提示很近的那批节点"正是我们要的归纳偏置。

    参考点只取**正点**（kind==0 & pol==0）与**负点**（kind==0 & pol==1，需 --pos-neg）；
    负属性/正属性是补充证据，不当参照点。参照点取的是**实际送进解码器的那份提示**，
    与 query_prior 同一条纪律（训练提示被抽稀过，用 q['query'] 会造成训练/测试不一致）。

    ---- 阶段37：`--seed-sim` 在几何块之后**再拼一块内容相似度**（默认 off ⇒ 逐位等价）----
    cos(H_v, mean_{正点} H) 与/或 max_{正点} cos(H_v, H_s)。它与几何场是**互补**的（离线实测：
    hop + 0.25·相似度 把 P@|A| 从 0.353 提到 0.400、配对 127胜/58负，且 unseen / 低重叠两组
    同样受益）；H 需从 decode() 传进来，且内部 **detach**（理由见 seed_sim_block 的 docstring）。
    宽度记账：本函数返回的最后一维 = `pos_dim_total(args)`（几何块 + 相似度块）。
    """
    if args.pos_mode == 'off' or not plists or data is None or args is None:
        return None
    dev = data['feat'].device
    # hop 走预计算距离表（每次 O(n)，便宜且无键空间）→ 只有含 ppr 的通道才需要缓存。
    use_neg = getattr(args, 'pos_neg', 'off') != 'off'
    use_cache = args.pos_chan in ('ppr', 'both') or (use_neg and args.pos_neg in ('ppr', 'both'))
    pzcache = data.setdefault('pos_cache', {}) if use_cache else {}
    # 缓存上限按**字节预算**折算成条数（一条 = n × pos_dim × 4 字节），约 400 MB 主机内存：
    # 训练提示每轮重抽子集 → 键空间会一直长（先验缓存实测到过 9.8 万条 / 3.1 GB）。
    pz_cap = max(1000, int(4e8 / max(1, data['n'] * pos_dim_total(args) * 4)))
    rows = []
    for pl in plists:
        pos = tuple(sorted({int(p['id']) for p in pl
                            if p['kind'] == 0 and p['pol'] == 0}))
        neg = tuple(sorted({int(p['id']) for p in pl
                            if p['kind'] == 0 and p['pol'] == 1})) if use_neg else ()
        # ⚠ 键必须带**负点元组与两个通道名**：只按正点做键的话，同一批正点在不同负点/不同
        #   通道下会拿回错的块（与 prompt_ball 的缓存键必须带 k 是同一类坑）。
        key = (pos, neg, args.pos_chan, getattr(args, 'pos_neg', 'off'))
        w = pzcache.get(key) if use_cache else None
        if w is None:
            if use_cache and len(pzcache) > pz_cap:
                # 纯 memoization：清掉只影响速度、不影响结果。
                pzcache.clear()
            w = torch.as_tensor(pos_feat(pos, neg, data, args, nodes), dtype=torch.float32)
            if use_cache:
                pzcache[key] = w
        # 阶段37：相似度块拼在几何块之后（不进缓存：见 seed_sim_block 的说明）
        sb = seed_sim_block(pos, H, args)
        if sb is not None:
            w = torch.cat([w, sb.to(w.device)], dim=1)
        rows.append(w)
    return torch.stack(rows).to(dev)               # (B,n,pos_dim)


# ==================== 属性通道（阶段45：--attr-mode sim）====================
def _attr_field_sparse(plists, data, nodes=None):
    """压缩属性（--attr-embed）下的属性场：用 CSR / 反向索引精确计算，不需要稠密 A。

    cos(A_v, q) 中 A_v 是二值属性行，q = 正属性均值 − 负属性均值再归一化：
        A_v·q = |S_v∩P|/|P| − |S_v∩N|/|N|
        ||A_v|| = sqrt(|S_v|)，||q|| = sqrt(1/|P| + 1/|N|)
    所以只需按属性反查节点并累加，复杂度是 O(命中节点数)，与属性维度 m 无关。
    之后同样做逐行中心化 + 标准化；无属性提示的整行仍为 0（保持"无属性=无通道"）。
    """
    import math
    dev = data['feat'].device
    n = data['n']
    rev_ptr = data['attr_rev_ptr']
    rev_nodes = data['attr_rev_nodes']
    cnt = np.maximum(np.asarray(data['attr_cnt'], dtype=np.float64), 1.0)
    sel = None if nodes is None else np.asarray(nodes, dtype=np.int64)
    nn = n if sel is None else int(sel.size)
    if sel is not None:
        pos_of = np.full(n, -1, dtype=np.int64)
        pos_of[sel] = np.arange(nn)
        cnt = cnt[sel]
    B = len(plists)
    F = np.zeros((B, nn), dtype=np.float32)
    for i, pl in enumerate(plists):
        pos = sorted({int(p['id']) for p in pl if p['kind'] == 1 and p['pol'] == 0})
        neg = sorted({int(p['id']) for p in pl if p['kind'] == 1 and p['pol'] == 1})
        if not pos and not neg:
            continue
        def _add(h, a, sign):
            v = rev_nodes[rev_ptr[a]:rev_ptr[a + 1]]
            if sel is not None:
                v = pos_of[v]
                v = v[v >= 0]
            h[v] += sign
        hit = np.zeros(nn, dtype=np.float64)
        for a in pos:
            _add(hit, a, 1.0)
        if pos:
            hit /= len(pos)
        if neg:
            h2 = np.zeros(nn, dtype=np.float64)
            for a in neg:
                _add(h2, a, 1.0)
            hit -= h2 / len(neg)
        qn = (1.0 / len(pos) if pos else 0.0) + (1.0 / len(neg) if neg else 0.0)
        F[i] = hit / (np.sqrt(cnt) * math.sqrt(qn))
    F = torch.from_numpy(F).to(dev)
    F = F - F.mean(dim=1, keepdim=True)
    F = F / F.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
    return F


def attr_field(plists, data, args, nodes=None):
    """提示的**属性 token** -> (B,n) 余弦相似度场：q = mean(正属性) − mean(负属性)。

    动机（cora_nosynth 的实测）：`attr`（纯属性提示）与 `full` 在 test 上给出**完全相同**
    的 F1 与掩码（seed 0/2 都是 0.360、seed 3 都是 0.001，8 个类别逐行相同）——说明属性
    token 只经由 two-way transformer 间接影响 query token，**从未进入"算掩码的那一步"**
    （logits = ⟨h_v, e⟩/√d）。而 cora/facebook 的类别/社区恰恰是词袋可分的（`attr` 行
    分别只有 0.047/0.024）。这里把"节点属性向量 × 查询属性向量"的余弦相似度**直接加到
    logits**，绕开 transformer 那条弱路径。

    定义：A = feat 的前 num_attrs 列（属性 ⊕ 结构里的属性部分），A_hat = 逐行 L2 归一化；
    q = 正属性均匀权重 − 负属性均匀权重，再归一化；sim = A_hat @ q ∈ [-1, 1]，
    然后**按查询中心化 + 标准化**（减行均值、除行标准差）。

    为什么要标准化（cora_attr 的实测）：原始余弦场大多落在 0.05–0.3（cora 平均每节点
    18.2 个词、查询 3 个词），`attr_w=1` 最多只抬高 0.3 个 logit —— 推不动 0.5 这条线；
    更糟的是余弦**非负**，只能加分、不能减分。实测 `n`（场=0）与 `n+a`（场≠0）的 F1
    完全相同（0.267 vs 0.263），`attr_w` 12 次都停在 1.00 ⇒ 场实际上是个 no-op。
    标准化之后场是零均值、单位方差：`attr_w=1` 等价于"1 个 logit 标准差"，且能加能减，
    属性相似度才可能真正改变 0.5 两侧的排序。

    ⚠ **无标签**：只用图自带的属性矩阵与提示本身 ⇒ 训练/测试同构。
    ⚠ 无属性 token 的提示（类别 `n` / `n-n`）→ 整行为 0（均值 0、标准差 0，除以 clamp
      后仍是 0），与不加该通道逐位相同 —— 这条不变式在标准化后依然成立。
    ⚠ **不缓存**：训练时每 batch 的提示集合几乎不重复，按 batch 建缓存会像 prior_cache
      那样把内存撑爆（历史上到过 3.1 GB）；这里每次重算，代价只是一次 B×m×n 的矩阵乘
      （n=3959、m=1406 时约 5.6M FLOP/查询，可忽略）。
    ⚠ 行归一化用的 A_hat 缓存在 `data['attr_hat']`（每图一次，n×m float32 ≈ 22 MB）。
    """
    if getattr(args, 'attr_mode', 'off') != 'sim' or not plists or data is None:
        return None
    if data.get('attr_ptr') is not None:
        return _attr_field_sparse(plists, data, nodes)
    dev = data['feat'].device
    m = int(data['num_attrs'])
    A = data.get('attr_hat')
    if A is None:
        X = data['feat'][:, :m]
        A = X / X.norm(dim=1, keepdim=True).clamp(min=1e-9)
        data['attr_hat'] = A
    Q = torch.zeros(len(plists), m, device=dev, dtype=A.dtype)
    for i, pl in enumerate(plists):
        pos = sorted({int(p['id']) for p in pl if p['kind'] == 1 and p['pol'] == 0})
        neg = sorted({int(p['id']) for p in pl if p['kind'] == 1 and p['pol'] == 1})
        if pos:
            Q[i, pos] += 1.0 / len(pos)
        if neg:
            Q[i, neg] -= 1.0 / len(neg)
    # 逐行归一化；全零行（无属性提示）保持 0 ⇒ 相似度场恒为 0。
    Q = Q / Q.norm(dim=1, keepdim=True).clamp(min=1e-9)
    F = Q @ A.t()                                    # (B,n) 余弦
    # 中心化 + 标准化：w=1 ⇔ "1 个 logit 标准差"，且允许负值（能压掉不相似的节点）。
    # 全零行 mean=0、std=0 ⇒ 0 / 1e-6 = 0，不破坏"无属性 = 无通道"。
    F = F - F.mean(dim=1, keepdim=True)
    F = F / F.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
    return F


def _cand_ball(seeds, data, hops, cap):
    """从正种子做 BFS，返回候选节点（按发现顺序，超过 cap 截断）。

    按 (seeds, hops, cap) 缓存：查询在 epoch 间固定，ball 不随训练变化，
    否则每个 epoch 都要重跑 2 千多次 Python BFS（实测占 8 分钟/epoch 的大头）。
    """
    key = (tuple(sorted(int(s) for s in seeds)), int(hops), int(cap))
    cache = data.setdefault('ball_cache', {})
    hit = cache.get(key)
    if hit is not None:
        return hit
    adj = data['adj']
    seen = set(int(s) for s in seeds)
    order, frontier = list(seen), list(seen)
    for _ in range(hops):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                v = int(v)
                if v not in seen:
                    seen.add(v); order.append(v); nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    if cap and len(order) > cap:
        order = order[:cap]
    out = torch.as_tensor(sorted(order), dtype=torch.long)
    if len(cache) < 20000:
        cache[key] = out
    return out


def _decode_ball(model, plists, x_prom, H, data, args):
    """批量局部解码：把一批提示的候选集补零到 max|C|，一次过解码器。

    需要 model.py 的 node_pad 支持（t2h 的 key_padding_mask），否则补零节点会被
    当成真节点参与注意力。pos / attr 仍按查询缓存（已切到 C），只做拼装。
    """
    dev = H.device
    n = H.size(0)
    hops = int(getattr(args, 'cand_hop', 2))
    cap = int(getattr(args, 'cand_cap', 15000))
    pc = data.setdefault('_pos_cache', {})
    ac = data.setdefault('_attr_cache', {})
    B = len(plists)
    Cs, pos_list, ab_list = [], [], []
    for pl in plists:
        seeds = [p['id'] for p in pl if p['kind'] == 0 and p['pol'] == 0]
        C = _cand_ball(seeds, data, hops, cap).to(dev)
        Cs.append(C)
        key = (tuple(sorted(int(s) for s in seeds)), hops, cap)
        pos_c = pc.get(key)
        if key not in pc:
            pos = query_pos([pl], data, args, H, nodes=C.cpu().numpy())
            pos_c = None if pos is None else pos
            if pos_c is not None and len(pc) < 8000:
                pc[key] = pos_c
        ab_c = attr_field([pl], data, args, nodes=C.cpu().numpy())
        pos_list.append(pos_c)
        ab_list.append(ab_c)
    L = max(int(c.numel()) for c in Cs)
    Hb = H.new_zeros(B, L, H.size(1))
    mask = torch.zeros(B, L, dtype=torch.bool, device=dev)
    for b, c in enumerate(Cs):
        Hb[b, :c.numel()] = H.index_select(0, c)
        mask[b, :c.numel()] = True
    pos_p = None
    if pos_list[0] is not None:
        pos_p = Hb.new_zeros(B, L, pos_list[0].size(-1))
        for b, p_ in enumerate(pos_list):
            pos_p[b, :p_.size(1)] = p_[0]
    ab_p = None
    if ab_list[0] is not None:
        ab_p = Hb.new_zeros(B, L)
        for b, a_ in enumerate(ab_list):
            ab_p[b, :a_.size(1)] = a_[0]
    ptok, pmask = model.prompt_enc(plists, x_prom)
    prior = query_prior(plists, data, args)
    if prior is not None:
        prior_p = Hb.new_zeros(B, prior.size(1), L)
        for b, c in enumerate(Cs):
            prior_p[b, :, :c.numel()] = prior[b, :, c]
        prior = prior_p
    lg, q = model.decoder(ptok, pmask, Hb, prior, pos_p,
                          attr_bias=ab_p, node_pad=~mask)
    full = torch.full((B, lg.size(1), n), -1e4, device=dev, dtype=lg.dtype)
    for b, c in enumerate(Cs):
        full[b, :, c] = lg[b, :, :c.numel()]
    return full, q


def _auto_cand(args, data):
    """用训练查询的 ball 覆盖率自动选口径（测试查询不参与，无泄漏）。"""
    tr = [q for q in data['qs'] if q.get('split') == 'train']
    if not tr:
        return 'full'
    covs = []
    for q in tr[:500]:
        seeds = [int(v) for v in q['query']['nodes']]
        gt = set(int(v) for v in q['target']['community_indices'])
        C = set(_cand_ball(seeds, data, int(args.cand_hop), 0).tolist())
        covs.append(len(gt & C) / max(1, len(gt)))
    m = sum(covs) / len(covs)
    mode = 'ball' if m >= args.cand_thresh else 'full'
    print(f'== 候选集自动选择 ==  train 覆盖率 mean {m:.3f}（{len(covs)} 条，r={args.cand_hop}）'
          f' → {mode}（阈值 {args.cand_thresh:.2f}）')
    return mode


def decode(model, plists, feat, H, data, args):
    """**统一的解码入口**：所有调用点都走这里。

    为什么要统一：先验与位置通道的口径（用哪份提示、正负锥怎么算、参考点是谁）必须训练与
    评估完全一致，靠"每个调用点记得传"是守不住的。--prior none / --pos-mode off 时各自返回
    None，解码器分别逐位等价于不带该通道的旧行为。

    ---- 阶段38：`--prompt-src {raw,H}` 决定**提示 token 的来源**（默认 raw ⇒ 逐位等价）----
    raw = 原始输入特征（`node_proj(feat[id])`，对齐 SAM 的模块分工：提示编码器与图编码器解耦）；
    H   = 图编码后的表示（`node_proj_h(H[id])`，提示侧并进图编码器，梯度从提示 token 回流）。
    "用哪个张量"必须和"用哪份提示"同一纪律 —— 放在这里统一裁决，调用方不各自决定。
    ---- 阶段45：`--attr-mode sim` 时再把 `attr_field` 的 (B,n) 属性余弦场传给解码器 ----
    ⚠ H 模式与 `--anon-seed-p > 0` 互斥（main 里 ap.error 拦）：匿名消融的前提正是
      "提示 token 保留原始信息、只堵编码器查节点身份"。
    """
    src = getattr(args, 'prompt_src', 'raw')
    if src == 'H':
        if H is None:
            raise ValueError('--prompt-src H 需要 H（编码器输出）；请从 decode() 调用并传入 H')
        x_prom = H
    else:
        x_prom = feat
    ptok, pmask = model.prompt_enc(plists, x_prom)
    ab = attr_field(plists, data, args)
    if ab is not None and model.training and getattr(args, 'attr_drop', 0.0) > 0:
        # ---- 属性场 dropout（--attr-drop，阶段46）----
        # 训练时按提示把**整行**场置零（推理时恒开）。动机：Cora 上开属性通道后 node-only 从
        # 0.245 掉到 0.070 —— 模型把决策整体让给了场，无属性提示的通路被饿死。按条抽掉场，
        # 等于强制同一模型在"有场/无场"两种条件下都要出掩码，两条通路因此都能保留。
        # 只按行抽（不按 batch）：同一批里保留一部分带场样本，避免整批退化。
        _keep = (torch.rand(ab.size(0), 1, device=ab.device) >= args.attr_drop).to(ab.dtype)
        ab = ab * _keep
    if getattr(args, 'cand', 'full') == 'ball' and H is not None:
        return _decode_ball(model, plists, x_prom, H, data, args)
    return model.decoder(ptok, pmask, H, query_prior(plists, data, args),
                         query_pos(plists, data, args, H),
                         attr_bias=ab)


def select_one(logits, quality):
    """从一次解码的 (B=1,K,n) + (1,K) 里按**部署口径**选一个候选：质量头 top-1（K=1 时唯一）。

    **单查询解码的调用点全都走这里**，不要再各写一遍选候选的规则 —— 那正是曾经的静默不一致：
    `split_f1`/`split_f1_groups` 写的是 `logits[0, 0]`（= 候选 0），而部署（`evaluate` 的
    `decode1`）发的是质量头 top-1。K=1 时两者恒等（所以所有 K=1 的实验都看不出来），
    **K>1 时早停就在优化另一个候选** —— 等于拿"没被选中的那个掩码"决定训练什么时候停。
    修法不是把那一处改成 `logits[0]`（那只是取回 (K,n)，还是候选 0），而是把规则收到这里。

    返回 (掩码 logits (n,), 全部候选的质量 (K,))。ACG 侧是同一规则的 numpy 版本
    （`q_all.argmax(axis=1)`），别在那里改成候选 0。
    """
    k = int(torch.argmax(quality[0])) if logits.size(1) > 1 else 0
    return logits[0, k], quality[0]


# ==================== 逐查询尺度（阶段20：--scale fit）====================
# 动机（阶段15/18 的诊断）：掩码规模是搜索侧最大的一档余量，而"该切多紧"**每条查询不同**
# —— 全局阈值不迁移（val 选出的最佳阈值在 seed 间 0.567±0.153，拿到 test 只值 +0.001）。
# 两个上界：oracle-T（每查询拿标签选最优档）= test +0.054~+0.075、oracle-size = +0.025~+0.036；
# 而"稳定性分数逐查询选档"这条**无标签规则已被否决**（它系统性选到更松的档）。
#
# 这里换成**学一个 T̂**：监督来自**训练查询**的 oracle-T（标签只在拟合时用），特征全是
# 测试时可算的**无标签**量（概率场的尺寸曲线 + 提示/邻域规模）→ 部署时它只是固定的离线函数。
# 判读必须**同 run 配对**（T=0 / T̂ / oracle-T 同屏）：T̂ ≈ T=0 → 学不动、放弃；
# T̂ 接近 oracle-T → 值得保留。
#
# ⚠ 这份实现被 `cam_simple.py` import 复用（那是最小模型）。**不要在两处各写一份** ——
#   `temp/verify_scale.py` 会检查全仓库只有一处定义。
SCALE_T = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)   # 档位在 logit 轴；0.0 ↔ p>0.5


def scale_features(plist, lg, data):
    """一条查询的**无标签**特征（测试时同样可算 —— 这是这条规则能部署的前提）。

    `lg` 是**已经选定的那一个候选**的逐节点 logits（形状 (n,)）—— 由调用方走 `select_one`
    取得。⚠ 这里**不许**自己写 `logits[0, 0]`：那是"取候选 0"，K>1 时就与部署口径（质量头
    top-1）不一致（阶段13 修过的同一类 bug，`verify_k_gran.py` 会拦）。
    ⚠ 也**不得**碰任何标签字段（`_full`）：`temp/verify_scale.py` 用 AST 钉死
    （查的是代码里的字符串常量、**不含 docstring** —— 本项目已 7 次栽在文本子串匹配上）。
    """
    z = lg.cpu().numpy()
    sizes = [int((z > t).sum()) for t in (0.0, 0.5, 1.0, 1.5)]
    pos_nodes = [int(p['id']) for p in plist if p['kind'] == 0 and p['pol'] == 0]
    pos_attrs = [p for p in plist if p['kind'] == 1 and p['pol'] == 0]
    ball = 0
    if pos_nodes:                 # 正点的 2 跳邻域规模（局部密度，也是"社区该多大"的代理）
        seen, fr = set(pos_nodes), set(pos_nodes)
        for _ in range(2):
            nx = set()
            for u in fr:
                nx.update(data['adj'][u])
            nx -= seen
            seen |= nx
            fr = nx
        ball = len(seen)
    deg = float(np.mean([len(data['adj'][u]) for u in pos_nodes])) if pos_nodes else 0.0
    l0 = np.log1p(sizes[0])
    return np.asarray([l0, np.log1p(sizes[1]), np.log1p(sizes[2]), np.log1p(sizes[3]),
                       np.log1p(sizes[1]) - l0, np.log1p(sizes[3]) - l0,   # 尺寸曲线的坡度
                       np.log1p(len(pos_nodes)), float(len(pos_attrs)),
                       np.log1p(ball), np.log1p(deg)], dtype=np.float64)


def oracle_T(lg, gt):
    """**训练期专用**（用标签）：档位里 F1 最好的那一档 —— 只作为拟合的监督。"""
    z = lg.cpu().numpy()
    best, bt = -1.0, 0.0
    for t in SCALE_T:
        f1 = prf(z > t, gt)[2]
        if f1 > best:
            best, bt = f1, t
    return bt


@torch.no_grad()
def fit_scale(model, data, args, device):
    """在**训练查询**上拟合 T̂ = w·φ(x) + b（岭回归；不重训 CAM，只多跑一遍训练集）。

    为什么用 train 而不是 val：val 是早停用的集合，拿它拟合等于把它偷偷变成训练集。
    n≈660、特征 10 维 ⇒ 岭回归足够稳。两条分支（原版/最小模型）共用这一份实现。
    """
    model.eval()
    H = model.encode(data['feat'], data['edge_index'])
    X, y = [], []
    with torch.no_grad():
        for q in data['splits']['train']:
            pl = make_prompt_noneg(q)
            lg_all, q_all = decode(model, [pl], data['feat'], H, data, args)
            lg, _ = select_one(lg_all, q_all)          # 与部署同一处选择规则
            X.append(scale_features(pl, lg, data))
            y.append(oracle_T(lg, q['_full'].cpu().numpy()))
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    A = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    lam = float(args.scale_lambda)
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return dict(w=w, mu=mu, sd=sd, n=len(X), lam=lam)


def apply_scale(fitv, feats):
    """特征 -> T̂（截到档位范围内）。"""
    t = float(((feats - fitv['mu']) / fitv['sd']) @ fitv['w'][:-1] + fitv['w'][-1])
    return float(np.clip(t, SCALE_T[0], SCALE_T[-1]))


@torch.no_grad()
def scale_report(model, data, args, device, fitv):
    """**同 run 内配对**：T=0 / T̂（学出来的）/ oracle-T（标签上界）三条口径的检索 + 检测。

    ⚠ 三者**平均化不同**：T=0 与 T̂ 是真正可部署的口径；oracle-T 是每查询用标签选出来的上界，
    只用来定标（"这条无标签规则吃掉了上界的多少"）。
    """
    model.eval()
    feats = data['feat']
    H = model.encode(feats, data['edge_index'])
    n2c = data['n2c']
    out = {}
    for tag, qs in (('test', data['splits']['test']),):   # val 只用于早停，评测不再跑
        f1 = [[], [], []]
        msz = [[], []]
        Ts = []
        grp = {g: [[], [], []] for g in ('overlap', 'plain')}
        for q in qs:
            pl = make_prompt_noneg(q)
            lg_all, q_all = decode(model, [pl], feats, H, data, args)
            lg, _ = select_one(lg_all, q_all)
            z = lg.cpu().numpy()
            gt = q['_full'].cpu().numpy()
            th = apply_scale(fitv, scale_features(pl, lg, data))
            preds = [shape_mask(z > t, data['adj'], args.shape)
                     for t in (0.0, th, oracle_T(lg, gt))]
            g = group_of(q, n2c)
            # ⚠ 三个口径各自一个槽（`grp[g][i]`）—— 首版写成 `grp[g][min(i, 1)]`，
            #   于是 i=1（T̂）与 i=2（oracle-T）挤进同一槽，打印出来的"T̂"其实是两者的平均
            #   （会把 oracle 的增益读成 T̂ 的）。守住在 temp/verify_scale.py。
            for i in range(3):
                f1[i].append(prf(preds[i], gt)[2])
                grp[g][i].append(prf(preds[i], gt)[2])
            msz[0].append(int(preds[0].sum()))
            msz[1].append(int(preds[1].sum()))
            Ts.append(th)
        m = lambda v: float(np.mean(v)) if v else 0.0            # noqa: E731
        out[tag] = dict(f1_t0=m(f1[0]), f1_th=m(f1[1]), f1_or=m(f1[2]),
                        msz_t0=m(msz[0]), msz_th=m(msz[1]),
                        T_mean=m(Ts), T_sd=float(np.std(Ts)),
                        T_zero=int(sum(1 for t in Ts if t == 0.0)), n=len(Ts),
                        ov_t0=m(grp['overlap'][0]), ov_th=m(grp['overlap'][1]),
                        ov_or=m(grp['overlap'][2]),
                        pl_t0=m(grp['plain'][0]), pl_th=m(grp['plain'][1]),
                        pl_or=m(grp['plain'][2]))
        o = out[tag]
        print('\n== 逐查询尺度  [%s]  n=%d ==' % (tag, o['n']))
        print('  T=0（p>0.5，现状）      F1 %.3f   掩码均值 %6.1f' % (o['f1_t0'], o['msz_t0']))
        print('  T̂（学出来的，无标签）   F1 %.3f   掩码均值 %6.1f   '
              'T̂ 均值 %+.2f±%.2f（=%d 条恰为 0）'
              % (o['f1_th'], o['msz_th'], o['T_mean'], o['T_sd'], o['T_zero']))
        print('  oracle-T（标签上界）    F1 %.3f' % o['f1_or'])
        print('  分组（overlap）：T=0 %.3f / T̂ %.3f / oracle %.3f'
              % (o['ov_t0'], o['ov_th'], o['ov_or']))
        print('  分组（  plain）：T=0 %.3f / T̂ %.3f / oracle %.3f'
              % (o['pl_t0'], o['pl_th'], o['pl_or']))
    print('  判读：T̂ ≈ T=0 → 学不动，放弃这条路；T̂ 与 oracle-T 的差 = 还剩多少。'
          '（三条口径同 run 配对，不受 run 噪声影响。）')

    # 检测侧同一件事：ACG 每个种子在 T=0 与 T̂ 下的 CLARE 口径（同 run 配对）
    # ⚠ 口径提示：拟合用的是**查询提示**，而 ACG 的提示是**单节点种子** —— 分布不完全同构，
    #   这一行只能当"同向证据"，不能当与查询侧同构的读数。
    n = data['n']
    comms = data['g']['communities']
    test_cids = {int(q['target']['community_id']) for q in data['splits']['test']}
    gt = {c: set(map(int, comms[c])) for c in test_cids if c in comms}
    gt_masks = [np.isin(np.arange(n), sorted(gt[c])) for c in sorted(gt)]
    gtn = set().union(*gt.values()) if gt else set()
    pools = {0: [], 1: []}
    # ⚠ 不要写成 `args.acg_batch`：run.py 没有这个字段，而 temp/check_smoke_args.py 会把
    #   "读了 argparse 没定义的 args.X"报出来（那是防止 AttributeError 的守卫，别绕过它）。
    _BATCH = 32
    for i in range(0, n, _BATCH):
        chunk = list(range(i, min(n, i + _BATCH)))
        plists = [[{'kind': 0, 'id': int(v), 'pol': 0}] for v in chunk]
        lg_all, q_all = decode(model, plists, feats, H, data, args)
        for k in range(lg_all.size(0)):
            lg, _ = select_one(lg_all[k:k + 1], q_all[k:k + 1])   # 同一处选择规则
            z = lg.cpu().numpy()
            th = apply_scale(fitv, scale_features(plists[k], lg, data))
            for which, t in ((0, 0.0), (1, th)):
                pred = shape_mask(z > t, data['adj'], args.shape)
                if pred.sum() >= 3:
                    pools[which].append(pred)
    print('\n== 检测 逐查询尺度（同 run 配对；提示分布与拟合时不完全同构）==')
    for which, name in ((0, 'T=0 '), (1, 'T̂   ')):
        ms = pools[which]
        if not ms:
            print('  %s 没有任何有效输出（跳过）' % name)
            out['acg_%d' % which] = None
            continue
        f1t = [max((prf(m, g)[2] for m in ms), default=0.0) for g in gt_masks]
        f1p = [max((prf(m, g)[2] for g in gt_masks), default=0.0) for m in ms]
        sym = 0.5 * (float(np.mean(f1t)) + float(np.mean(f1p)))
        un = set().union(*[set(np.nonzero(m)[0].tolist()) for m in ms])
        print('  %s  AvgF1(对称) %.3f  AvgF1(真值侧) %.3f  found %d/%d  detect%% %.3f  输出 %d'
              % (name, sym, float(np.mean(f1t)), int(sum(f > 0.5 for f in f1t)),
                 len(gt_masks), len(gtn & un) / len(gtn) if gtn else 0.0, len(ms)))
        out['acg_%d' % which] = dict(sym=sym, true=float(np.mean(f1t)),
                                     found=int(sum(f > 0.5 for f in f1t)),
                                     detect=len(gtn & un) / len(gtn) if gtn else 0.0)
    return out


# ==================== 训练 ====================
def select_epoch_queries(real_qs, synth_qs, synth_per_real, rng):
    """每轮保留全部真实查询，并无放回抽取指定比例的合成查询。"""
    if synth_per_real is None:
        selected_synth = list(synth_qs)
    else:
        n_synth = min(len(synth_qs), int(len(real_qs) * synth_per_real + 0.5))
        selected_synth = rng.sample(synth_qs, n_synth)
    return list(real_qs) + selected_synth


def strip_neg(plist):
    """剥掉提示里的**全部负 token**（负点与负属性），只留正点/正属性。

    用途（阶段17，`--drop-neg-train`）：训练侧**每一处**提示构造都必须过这道闸门，否则
    "不考虑负提示"这个消融不成立。首版只在多轮循环里剥，**漏了属性辅助任务那条支路**
    （`--attr-aux` 有自己的 `decode`、且在循环之外），于是负属性仍以 0.1 的权重进模型 ——
    也就是说那一跑的准确口径是"主路径零负提示 + 负属性辅助支路"，不是"零负提示"。

    刻意**保留正属性**：`--attr-aux` 问的是"属性提示能不能定位社区"，与"要不要负提示"
    是两个问题，一起动就不是单变量。

    ⚠ 这里**不给空结果兜底**：调用方自己决定（主分支历史上写的是 `strip_neg(pl) or pl`，
    即"整条提示都是负的"时原样保留；属性辅助支路则应当直接丢弃该查询）。
    """
    return [p for p in plist if p['pol'] == 0]


def sample_train_prompt(q, mode, p_drop, rng, neg_keep='mix', attr_only_p=0.0,
                        cats=False, neg_cover=2.0 / 3.0):
    """构造首轮提示；mix 显式覆盖部署检索和 ACG 的提示形态。

    `cats=True`（阶段31，`--prompt-cats taxonomy`）＝ 把形态换成**8 个提示类别**：
    `n` / `n+a` / `n-n` / `n-a` / `n+a-n` / `n+a-a` / `n-n-a` / `full`
    （`n`=正节点必备，其余三位是 {正属性, 负节点, 负属性} 的 7 个非空子集），
    权重由 `neg_cover` 决定（无负类别共 1-neg_cover，带负类别共 neg_cover）。

    ⚠ 三条与旧模式的差别（都是刻意的）：
      1. **不做抽稀**：`p_drop` 与 `--prompt-mode dropout` 在类别模式下**不生效** ——
         抽稀会让"标着 n+a-n 的提示"丢掉属性，类别就不再是类别；多样性改由 8 个类别
         本身提供。
      2. **`--neg-keep` 不生效**：类别模式里"带不带负"由类别决定，`all` 会把无负类别
         也补上负点，等于把类别表毁掉（main 里对这两个组合有 ap.error 拦）。
      3. `--attr-only-p`（纯属性形态）与"正节点必备"冲突，故在类别模式下**默认关**；
         纯属性这条训练信号请用 `--attr-aux`（它本来就在教"属性→圈掩码"）。
    ⚠ 兜底：若某条查询**没有正节点**（数据侧不该出现），退回 `full` 并记
      `fallback_no_pos_node`（会出现在 epoch 行的 prompt forms 里，正常应为 0）。

    ---- 下面是 legacy 模式（`--prompt-cats legacy`）的原有说明 ----
    `neg_keep`（阶段16，`--neg-keep`）只管**负点的保留率**：
      · 'mix'（默认）= 现状：负点只在 'full' 形态（约 1/4）与 'subset' 侥幸存活时出现，
        实测只有约 1/3 的训练提示带负点 —— 而**部署时每条提示都带**（`make_prompt(q)`
        恒给负点）。于是模型被训练成"多数时候忽略负点"，这与实测吻合：负提示增益 all 仅
        +0.001、把负点做成解码器几何通道（阶段14）也无效、① B\\A 始终占 FP 质量 ~51%。
      · 'all' = **任何形态都补回该查询的全部负点**，负点覆盖≈100%，与部署口径一致。
        这是"① 不降到底是**覆盖率**问题还是模型**学不会**"的单变量实验。

    ⚠ 三条不变式（由 temp/verify_neg_keep.py 钉死）：
      1. 负点是**后拼**的：只加不减、不改顺序（已存在的负点原样保留，只补缺的）；
      2. **不动正点**：同一 rng 流下 'mix' 与 'all' 的正点侧逐位相同；
      3. **不动随机流**：'all' 不额外消耗 rng（因此两个设置的 rng 轨迹完全一致）。

    `attr_only_p`（阶段23，`--attr-only-p`）＝ 以该概率把提示换成**纯属性形态**（'attr'）：
    去掉**全部节点 token**、只留属性 token（正 + 负）。为什么（output68 的 `attr` 行）：
    纯属性提示实测 F1 **0.049**、掩码均值 **3.5** 个节点（GT 53.6），而 base 0.314 ——
    检索分数**全部**来自种子点；ACG 侧 n+a 也只比 node-only 高 +0.006。而查询的属性取自
    `community_attributes`（社区 top-3、共享属性优先）⇒ 那是**社区级语义画像**，也是唯一能
    迁移到"训练集里没有的社区"的输入。

    ⚠ **不要写成"模型从没见过纯属性提示"**（我首版就是这么写的，错的）：`train()` 里本来
      就有一条属性-only 辅助任务（`--attr-aux`，默认 0.1），它对**每条查询、每个 epoch**
      都算一次"纯属性 → 圈掩码"的损失，构造与本形态**完全相同**（都是 `make_prompt` 里
      `kind == 1` 的过滤）。所以本开关不是"第一次让模型见到纯属性提示"，而是把同一条提示
      放进**主任务**（该样本权重 1.0 而非 0.1）、并让它**进入交互轮**（第 2 轮起会被补上
      错误驱动点）—— 力度更大、且与部署口径同构。想更轻地先试，直接抬 `--attr-aux` 即可
      （已有开关、零代码改动）；两者读数都看检索表的 `attr` 行。
      机制上的候选解释因此是**权重比**（主任务 1.0 可以把属性支路压在 0.1 上）+ 捷径学习，
      而不是"没有信号"。

    ⚠ **默认 0.0 时一个随机数都不抽**（`attr_only_p > 0 and ...` 短路）⇒ 逐位复现旧行为。
      这一族 bug 的通用形态就是"默认关闭的开关偷偷改了随机流"，守卫在 temp/verify_attr.py。
    ⚠ 该形态**不接 `_add_neg`**：负**节点** token 会把"无节点"这个前提破坏掉；
      负**属性**已随 `kind == 1` 一起带上了。配 `--drop-neg-train` 时只剩正属性
      （而评测端的 `attr` 行仍带负属性，与 +neg/base 行的部署口径一致）。
    ⚠ 查询**没有正属性**时不换形态（`and has_pos_attr`；纯负属性的提示没有可匹配的语义
      画像，等价于空提示）。
    ⚠ 随机流的性质（说准确）：`attr_only_p = 0` 时**一个随机数都不抽**；`> 0` 时形态开关
      恒为**该次调用的第一个抽取** —— 所以"是否切换"只由这一个随机数决定、与查询内容和
      后续形态无关；切换失败（无正属性）后走的仍是原来的形态分支，抽取自然与旧版一致。
      （首版 docstring 写成"随机流只取决于 attr_only_p、与查询内容无关"，那是错的：
      回落路径的抽数取决于原来抽到哪个形态。）
    """
    full = make_prompt(q)
    negs = [p for p in full if p['pol'] == 1]
    attrs_only = [p for p in full if p['kind'] == 1]        # 纯属性形态的候选（正 + 负属性）
    has_pos_attr = any(p['pol'] == 0 for p in attrs_only)

    # 阶段23：按概率换成纯属性形态。抽取放在 `attr_only_p > 0` 之后 ⇒ 默认 0 时不改变随机流；
    # `has_pos_attr` 是必要条件（纯负属性的提示没有可匹配的语义画像，等于退化成空提示）。
    if attr_only_p > 0 and rng.random() < attr_only_p and has_pos_attr:
        return attrs_only, 'attr'

    # ---- 阶段31：8 类别形态（正节点必备；带负类别由 neg_cover 控制）----
    if cats:
        cat = rng.choices(PROMPT_CATS, weights=cat_weights(neg_cover), k=1)[0]
        pl = make_prompt_cat(q, cat)
        if not has_pos_node(pl):
            # 兜底：无正节点的查询（数据侧不该有）退回 full，并在 prompt forms 里可见
            return full, 'fallback_no_pos_node'
        return pl, cat

    def _add_neg(plist):
        """只补**缺的**负点（幂等）：已带的原样留在原位，顺序不变。"""
        if neg_keep != 'all' or not negs:
            return plist
        have = {(p['kind'], p['id']) for p in plist if p['pol'] == 1}
        return plist + [p for p in negs if (p['kind'], p['id']) not in have]

    if mode == 'dropout':
        return _add_neg([p for p in full if rng.random() >= p_drop]), 'dropout'

    form = rng.choice(('full', 'positive', 'seed', 'subset'))
    if form == 'full':
        return _add_neg(full), form          # full 已含全部负点 → _add_neg 原样返回

    positive = [p for p in full if p['pol'] == 0]
    if form == 'positive':
        return _add_neg(positive or full), form

    if form == 'seed':
        nodes = [p for p in positive if p['kind'] == 0]
        attrs = [p for p in positive if p['kind'] == 1]
        seed = rng.choice(nodes or positive or full) if full else None
        prompt = [seed] if seed is not None else []
        if nodes and attrs and rng.random() < 0.5:
            prompt += rng.sample(attrs, rng.randint(1, min(3, len(attrs))))
        return _add_neg(prompt), form

    # 随机子集仍提供提示多样性，但保证至少保留一个正提示。
    prompt = [p for p in full if rng.random() >= p_drop]
    if positive and not any(p['pol'] == 0 for p in prompt):
        prompt.append(rng.choice(positive))
    return _add_neg(prompt), form


def utils_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / union if union else 0.0


def resolve_pw_max(loss_kind, pw_max):
    """pos_weight 上限的默认值随损失类型而定。

    `bce_dice` 保持旧口径 50（**逐位复现历史日志**）；`tversky` 默认 1（中性），
    因为 tversky 的用意正是去掉"漏检代价是误检 50 倍"这个召回偏置。
    显式传 `--pw-max` 则一律以显式值为准，用来做单变量消融。
    """
    if pw_max is not None:
        return float(pw_max)
    return 50.0 if loss_kind == 'bce_dice' else 1.0


def alt_neg_mask(q, device):
    """种子点的**其它社区（B）去掉自身社区（A）**之后的节点掩码 —— 只给阶段24 的 B\\A 加压项用。

    为什么有它：假阳性质量分解里 `种子点其它社区(B\\A)` 一路稳定占 **FP 质量 48~49.5%**
    （三次输入侧机制——加性先验、负参照几何通道、负点覆盖率——都没动过它），而塌缩诊断显示
    **84~85% 的最佳匹配落在训练社区** ⇒ 模型现在的答案就是"种子点所属的那个训练社区"。
    `q['_alts']`（`build_alt_targets`）恰好是 `[A] + 种子点的其它社区`，且已被 `[防泄漏]`
    限制在 train 划分内 ⇒ 训练时直接就有了 B，不需要任何新标签。

    ⚠ **必须减掉 A**：种子点是重叠点，B 与 A 在成员上可以相交（社区重叠），不去交集的话这一项
      会惩罚真阳性 —— 这是本项唯一危险的失败模式，守卫见 temp/verify_altneg.py 与冒烟。
    ⚠ 候选数 <2 的查询（plain 组）没有 B，返回 None（调用方跳过）。
    """
    alts = q.get('_alts') or []
    if len(alts) < 2:
        return None
    A = q['_full'].to(device).bool()
    hat = torch.zeros_like(A)
    for m in alts[1:]:
        hat = hat | m.to(device).bool()
    return hat & ~A


def anon_seed_rows(batch, p, rng, n):
    """阶段27（路 A）：抽"这条查询的**正种子点**行要在**编码器输入**里被匿名（置零）"。

    动机（阶段26 的结构性发现）：假阳性的两个成分（**训练社区** / 种子点**其它社区 B\\A**）
    会互相顶替，**合计守恒在 FP 质量的 75~84%**（15 次运行）⇒ "只削一个成分"的机制（输入侧喂
    信息、输出侧选择性惩罚、增广打记忆）注定只换来份额重排。而**捷径本身**是：编码器读得出
    "这是哪个节点" ⇒ 背出它所属的训练社区（84~85% 的最佳匹配落在训练社区）。本开关堵的就是这条：
    编码器看到的是**种子点特征被置零**的图，而**提示通道仍用原始特征**（调用方给 `decode` 的仍是
    `feat_t`）⇒ "用户点了哪个点"完整保留，"这个点的身份"不再可查。

    ⚠ 只匿名**正种子点**（`q['query']['nodes']`）：负点是**提示信息**（用户指定的"不是它"），
      不是捷径，保持原样。
    ⚠ 与 `--feat-drop` 的区别：那是**全局**随机丢特征（阶段26 实测零代价但没堵住捷径），这是
      **针对种子点**的定向版；也与 `--hide-ratio`（整社区不进 batch）不同：数据没少，只是身份不可读。
    ⚠ **训练期正则，评测期不匿名**（部署输入不做手脚），与 feat_drop 同构。
    ⚠ 默认 0.0 ⇒ **一个随机数都不抽**（`p > 0` 在最左短路）⇒ 逐位等价旧行为。
    ⚠ 交互轮的错误驱动补点节点**不在**匿名集合里（那些点第 1 轮不存在、部署时也不存在，不构成捷径）。

    返回 `(rows, n_q)`：`rows` 是 (n,) 的 CPU bool（None = 本次没有匿名任何节点），
    `n_q` = 至少被匿名一个种子的查询数（给日志的 `匿名 x/y` 读数用）。
    """
    if p <= 0 or not batch:
        return None, 0
    hit = set()
    n_q = 0
    for q in batch:
        ids = ((q.get('query') or {}).get('nodes'))
        ids = (ids.tolist() if hasattr(ids, 'tolist') else list(ids or []))
        if not ids:
            continue
        if rng.random() < p:
            hit.update(int(v) for v in ids)
            n_q += 1
    if not hit:
        return None, 0
    rows = torch.zeros(n, dtype=torch.bool)
    rows[torch.as_tensor(sorted(hit), dtype=torch.long)] = True
    return rows, n_q


def compute_loss(logits, quality, batch, device, loss_kind='bce_dice',
                 pw_max=50.0, tv_alpha=0.7, tv_beta=0.3, slot_targets=None,
                 alt_neg_w=0.0):
    """K 粒度损失：每个**槽位**对各自的候选目标算 BCE(pos_weight) + {dice|tversky}，
    外加质量头 MSE —— 质量头的目标是"**该候选 vs 查询真值 A** 的 IoU"。

    ⚠ 质量头目标刻意用 A 而不是候选自己的目标（方案 4 的关键）：这样质量头学的是
    "这个候选是不是被问的那个社区"，它才是 A/B 歧义的选择器。代表混淆社区 B 的候选
    对 A 的 IoU 很低 → 被学成低分；而质量头的输入 `e_k` 经 two-way transformer 被
    **提示（含负点）条件化**过，所以它能学会"提示说 −b 时给 A 那个候选高分"。

    `slot_targets`：长度 B 的列表，每项是长度 K 的列表，元素为候选目标张量或 None（padding）。
    给 None 时走旧路径（K=1 且用 q['_full']）→ **`--k-gran 1` 与旧实现逐位等价**。

    ⚠ 槽位顺序由调用方（train）**每 epoch 打乱**：若把 A 固定放槽 0，模型就能"永远输出槽 0"、
    根本不需要靠提示消歧，机制不会被逼出来。规范序只在 `q['_alts']` 里保存，见 build_alt_targets。

    `alt_neg_w`（阶段24，`--alt-neg-w`）：对**种子点的其它社区 B 减去自身社区 A** 的那些节点
    额外加一项 `w · BCE(logits → 0)`（B\\A 与 A 恒不相交，故不会惩罚真阳性）。

    为什么是它：B\\A 稳定占 **FP 质量 48~49.5%**（三次输入侧机制——加性先验、负参照几何通道、
    负点覆盖率——都没动过它），而"完美清掉 B\\A 且 TP 不变"的算术上界是检索 **0.316→0.48
    （+0.16）**。与已失败的两类机制的区别：
      · 输入侧喂信息 → 模型拿不到、也没被逼用；
      · 全局收紧（pw_max 加码）→ 不分对象地压所有负类；
      本项是**输出侧的选择性监督**：只对"这是种子点的另一个社区"那些节点加压。实现上等价于
      把 B\\A 节点的负类权重从 1 提到 1+w，其余一字不动。

    ⚠ 只加在**目标就是 A** 的槽位上（K=1 时 `slots is None` 天然如此；K>1 时按
      `torch.equal(tgt, A)` 判定）—— K>1 的候选里本来就含 B 的候选，那个槽位**应当**输出 B，
      对它加压等于自相矛盾。
    ⚠ 默认 0.0 ⇒ 整项被跳过（不抽随机数、不额外前向）⇒ 逐位等价旧行为。
    ⚠ 属性辅助支路共用同一个 `loss_fn` ⇒ 这一项**也作用在它上面**（再乘 `attr_aux` 的权重）。
      这是刻意的：那条支路的目标同样是 A，同样不该输出 B；影响量级 = attr_aux（默认 0.1）倍。
    """
    K = logits.size(1)
    loss = 0.0
    tgt_io = []
    for j, q in enumerate(batch):
        A = q['_full'].to(device)                     # 查询真值，用于质量头目标
        slots = None if slot_targets is None else slot_targets[j]
        row = []
        for k in range(K):
            if slots is not None and k < len(slots):
                tgt = slots[k]
                if tgt is None:                       # 候选不足的 padding 槽位：跳过
                    row.append(0.0)
                    continue
                tgt = tgt.to(device)
            elif slots is None and K == 1:
                tgt = A                               # 旧路径（--k-gran 1）
            else:
                row.append(0.0)                       # 旧路径但 K>1：无候选信息，屏蔽
                continue
            prob = logits[j, k].sigmoid()
            pw = ((~tgt).sum() / tgt.sum().clamp(min=1)).clamp(max=pw_max)
            loss = loss + F.binary_cross_entropy_with_logits(
                logits[j, k], tgt.float(), pos_weight=pw)
            if loss_kind == 'tversky':
                loss = loss + tversky_loss(prob, tgt, tv_alpha, tv_beta)
            else:
                loss = loss + dice_loss(prob, tgt)
            # ---- 阶段24：B\\A 选择性抑制（--alt-neg-w；默认 0 时整块跳过）----
            # 只对"目标就是 A"的槽位加压：K>1 的候选里含 B 的候选，那个槽位本该输出 B。
            if alt_neg_w > 0 and (slots is None or torch.equal(tgt, A)):
                hat = alt_neg_mask(q, device)
                if hat is not None:
                    sel = logits[j, k][hat]
                    if sel.numel():        # 空掩码会让 BCE 变成 nan（0/0）
                        loss = loss + alt_neg_w * F.binary_cross_entropy_with_logits(
                            sel, torch.zeros_like(sel))
            row.append(utils_iou(
                (logits[j, k].detach().sigmoid() > 0.5).cpu().numpy(),
                A.cpu().numpy()))
        tgt_io.append(row)
    tgt_io = torch.as_tensor(tgt_io, dtype=torch.float32,
                             device=device).reshape(len(batch), K)
    return loss + F.mse_loss(quality, tgt_io)


def slot_targets_of(batch, k_gran, rng):
    """按**每 epoch 每查询随机打乱**的方式，把候选填进 K 个槽位（不足补 None）。

    为什么要打乱：候选的规范序是"A 在前"（build_alt_targets）。若训练时也固定这个顺序，
    模型可以只靠槽位身份作答、完全不必看提示，质量头也学不到消歧 —— 那方案 4 就白做了。
    打乱后，哪个槽位是 A 每轮都在变，唯一稳定的判据只剩提示本身。
    `--k-gran 1` 时退化为 [[A]]，与旧实现逐位一致。
    """
    out = []
    for q in batch:
        alts = q.get('_alts')
        # ⚠ 缺 `_alts` 的查询会让它的 K 个槽位**全部**变成 padding（= 该查询一点分割梯度都没有），
        # 而且不会报错 —— 这是静默降级。首跑就踩了：合成圈的查询字典由 make_synth_queries
        # 自己构造、不带 `_alts`，于是每轮约 11% 的 batch（80 条合成查询）白跑。
        # 所以这里断言而不是跳过：宁可在第一个 batch 就炸，也不要悄悄训出一个不干净的模型。
        assert alts is not None, (
            '训练池里有查询缺 _alts（合成圈？main 里生成 synth_qs 后要补 _alts）',
            q.get('synth'), int(q['target']['community_id']))
        alts = alts[:k_gran]
        if len(alts) > 1:
            order = rng.sample(range(len(alts)), len(alts))
            slots = [alts[i] for i in order]
        else:
            slots = list(alts)
        slots += [None] * (k_gran - len(slots))
        out.append(slots)
    return out


@torch.no_grad()
def split_f1(model, H, split_qs, feat, adj=None, shape='lcc', data=None, args=None):
    model.eval()
    f1s = []
    for q in split_qs:
        # 选候选必须走 select_one（部署口径 = 质量头 top-1）。以前这里写的是 logits[0, 0]，
        # K>1 时早停评的是候选 0、部署发的是 top-1，两者不是同一个东西。
        lg, _ = select_one(*decode(model, [make_prompt(q)], feat, H, data, args))
        pred = shape_mask((lg.sigmoid() > 0.5).cpu().numpy(), adj, shape)
        f1s.append(prf(pred, q['_full'].cpu().numpy())[2])
    return float(np.mean(f1s))


def group_of(q, n2c):
    """查询分组：种子点落在重叠社区 = overlap，否则 plain（与 test 检索表同口径）。"""
    seeds = q['query']['nodes'].tolist()
    return 'overlap' if any(len(n2c.get(int(v), [])) >= 2 for v in seeds) else 'plain'


@torch.no_grad()
def traj_stats(model, data, args, qs, H, want_plain=True):
    """**轨迹诊断**用的读数：三条口径的 F1 + 掩码均值 + 真覆盖中位 + plain 组。

    动机（RUN-LIST 阶段21/22）：所有配置的最终数字都取自 epoch 2~11 的快照，而 val 先冲高
    后回落。output66 的轨迹把成因定住了：**峰值在 epoch 4、test 也同意**（所以不是选点坏了），
    之后 val/test 一起下滑、而**掩码均值从 ~51 震荡上行到 ~167（GT 53.6）** → 是"过覆盖回潮"。

    本函数因此报**三条口径**，用来判定"下滑是否纯尺度"：

      · `f1`        T=0（`sigmoid(logit) > 0.5`，与 `split_f1`/主表"阈值 0.5"**同一口径**）
      · `f1_mean`   **mean 切**：阈值放在**该查询自己字段的均值**上 —— 无标签、逐查询，
                    正是"把 logits 逐查询标准化"在评测端的等价物（`logit > mean` ⇔ 标准化后 `> 0`）。
                    **若它在后期 epoch 上不跟着掉 → 漂移不变切有效**，值得实现；
                    若跟着掉 → 排序本身退化，该走反记忆而不是尺度。
      · `f1_ors`    oracle-size（按 |GT| 切，标签上界）：它不掉而 T=0 掉 = 排序还完好、只是尺度跑了。

    ⚠ **只诊断**（`f1_ors` 用标签、只打印）：调用点只在 `train()` 里那一个块；
    `temp/verify_traj.py` 会检查"全部调用点都在块内、块内不写 best_f1/best_state/bad、
    读数只进 print"。口径用 `make_prompt`（+neg）与早停 `split_f1`、主表 `test +neg` 行一致
    —— 所以 val 那行的 `f1` 应与 epoch 行里的 `val F1` 相等（两条代码路径，可作自检）。
    """
    f1s, msz, gen, pl = [], [], [], []
    f1m, mszm, f1o = [], [], []
    for q in qs:
        plist = make_prompt(q)
        lg, _ = select_one(*decode(model, [plist], data['feat'], H, data, args))
        z = lg.cpu().numpy()
        gt = q['_full'].cpu().numpy()
        # ⚠ 阈值口径**必须**是 `sigmoid(z) > 0.5`（⇔ z > 0），与 `split_f1` / `mask_of`
        #   （主表"阈值 0.5"）逐位一致；首版写的是 `z > 0.5`（= p > 0.622，logit 空间），
        #   于是这一列与 epoch 行的 val F1 不等（output67 里 e2 0.031 vs 0.086、e3 0.332 vs
        #   0.322），我加的那句"[epoch 行 val F1，应相等]"自检当场就把它抓出来了。
        #   `z` 本身（未过 sigmoid）留给 mean 切与 oracle-size：两者都只依赖单调性/排序。
        pred = shape_mask(lg.sigmoid().cpu().numpy() > 0.5, data['adj'], args.shape)
        f1 = prf(pred, gt)[2]
        f1s.append(f1)
        msz.append(int(pred.sum()))
        gen.append(int(np.logical_and(pred, gt).sum()))
        if want_plain and group_of(q, data['n2c']) == 'plain':
            pl.append(f1)
        # mean 切（无标签、逐查询）：漂移不变性的探针
        pred_m = shape_mask(z > float(z.mean()), data['adj'], args.shape)
        f1m.append(prf(pred_m, gt)[2])
        mszm.append(int(pred_m.sum()))
        # oracle-size（标签上界）：排序是否还完好
        k = int(gt.sum())
        pred_o = np.zeros_like(gt)
        if k:
            pred_o[np.argpartition(-z, k - 1)[:k]] = True
        f1o.append(prf(pred_o, gt)[2])
    _m = lambda v: float(np.mean(v)) if v else 0.0                 # noqa: E731
    return dict(n=len(f1s), f1=_m(f1s), msz=_m(msz), gen_med=float(np.median(gen)) if gen else 0.0,
                plain=(_m(pl) if pl else 0.0), n_plain=len(pl),
                f1_mean=_m(f1m), msz_mean=_m(mszm), f1_ors=_m(f1o))


def covered_nodes(data, split='train'):
    """被该 split 的社区覆盖的节点集合（种子覆盖口径）。"""
    cids = {int(q['target']['community_id']) for q in data['splits'][split]}
    out = set()
    for c, m in data['g']['communities'].items():
        if int(c) in cids:
            out.update(int(v) for v in m)
    return out


def exposure_name(q, covered):
    """种子点是否落在训练社区覆盖的节点上：seen / unseen。

    这是与社区级划分正交的一条轴：即便测试社区未见，只要它的种子节点被某个
    训练社区覆盖，模型仍可能靠记忆作答。检索/ACG 的泛化口径都应分开报。
    """
    seeds = q['query']['nodes'].tolist()
    return 'seen' if any(int(v) in covered for v in seeds) else 'unseen'


def community_exposure(data, split):
    """{社区 id: 与训练社区成员的最大 Jaccard}——成员重叠度（社区标签本身未见）。"""
    comm_members = {int(c): set(map(int, m))
                    for c, m in data['g']['communities'].items()}
    train_cids = {int(q['target']['community_id']) for q in data['splits']['train']}
    train_sets = [comm_members[c] for c in train_cids if c in comm_members]
    out = {}
    for q in data['splits'][split]:
        cid = int(q['target']['community_id'])
        if cid not in out:
            s = comm_members.get(cid, set())
            out[cid] = max((len(s & t) / len(s | t) for t in train_sets), default=0.0)
    return out


@torch.no_grad()
def split_f1_groups(model, H, split_qs, feat, n2c, covered=None, expo_of=None,
                    adj=None, shape='lcc', data=None, args=None):
    """分组的 F1（用于检查 val 与 test 的构成偏差：早停准不准取决于此）。"""
    model.eval()
    acc = {'all': [], 'overlap': [], 'plain': []}
    if covered is not None:
        acc['seen'] = []
        acc['unseen'] = []
    if expo_of is not None:
        acc['重叠≥0.3'] = []
        acc['重叠<0.3'] = []
    for q in split_qs:
        # 同 split_f1：走 select_one（部署口径），别用候选 0。
        lg, _ = select_one(*decode(model, [make_prompt(q)], feat, H, data, args))
        pred = shape_mask((lg.sigmoid() > 0.5).cpu().numpy(), adj, shape)
        f1 = prf(pred, q['_full'].cpu().numpy())[2]
        acc['all'].append(f1)
        acc[group_of(q, n2c)].append(f1)
        if covered is not None:
            acc[exposure_name(q, covered)].append(f1)
        if expo_of is not None:
            cid = int(q['target']['community_id'])
            acc['重叠≥0.3' if expo_of.get(cid, 0.0) >= 0.3 else '重叠<0.3'].append(f1)
    return {g: (float(np.mean(v)) if v else 0.0, len(v)) for g, v in acc.items()}


def split_audit(data):
    """val/test 可比性核对：早停用 val，所以两者必须在这些维度上可比。

    逐项：提示构成（正/负点与属性的平均条数、无属性提示的查询数）、
    目标圈结构（规模、GT conductance）、种子覆盖（种子点是否落在训练社区节点内）。
    任何一项差异过大，val 选出的 epoch 就不代表 test 的表现。
    """
    ei, n = data['ei_np'], data['n']
    comms = {int(c): set(map(int, m)) for c, m in data['g']['communities'].items()}
    train_cids = {int(q['target']['community_id']) for q in data['splits']['train']}
    train_sets = [comms[c] for c in train_cids]
    covered = covered_nodes(data, 'train')
    print('== val/test 可比性核对（早停代理是否有效）==')
    for name in ('val', 'test'):
        qs_ = data['splits'][name]
        cids_ = {int(q['target']['community_id']) for q in qs_}
        cnt = Counter(group_of(q, data['n2c']) for q in qs_)
        seen = sum(any(int(v) in covered for v in q['query']['nodes'].tolist())
                   for q in qs_)
        no_attr = sum(1 for q in qs_ if len(q['query']['attrs']) == 0)
        posn = np.mean([len(q['query']['nodes']) for q in qs_])
        posa = np.mean([len(q['query']['attrs']) for q in qs_])
        negn = np.mean([len(q['query']['neg_nodes']) for q in qs_])
        nega = np.mean([len(q['query']['neg_attrs']) for q in qs_])
        sizes = [len(comms[c]) for c in cids_]
        conds = [conductance(ei, q['_full'].numpy(), n) for q in qs_]
        expo = [max((len(comms[c] & t) / len(comms[c] | t) for t in train_sets),
                    default=0.0) for c in cids_]
        print(f'  {name:4s} | overlap {cnt["overlap"]}/plain {cnt["plain"]}；'
              f'种子落在训练社区 {seen} / 未覆盖 {len(qs_) - seen}；'
              f'提示：正点 {posn:.2f} 正属性 {posa:.2f}，无属性提示 {no_attr}，'
              f'负点 {negn:.2f} 负属性 {nega:.2f}')
        print(f'       | 目标圈：规模中位 {np.median(sizes):.0f} '
              f'GT conductance {np.mean(conds):.3f}（越小越紧凑）；'
              f'与训练圈成员重叠 均值 {np.mean(expo):.2f} 最大 {np.max(expo):.2f} '
              f'（≥0.3：{sum(e >= 0.3 for e in expo)}/{len(expo)}）')


def train(model, data, args, device):
    """多轮交互训练（--rounds，默认 2）；早停参数在内存中回载（不落盘）。"""
    feat, edge_index = data['feat'], data['edge_index']
    rng = random.Random(args.seed)
    sample_rng = random.Random(args.seed * 7919 + 17)
    prompt_rng = random.Random(args.seed * 104729 + 29)
    real_qs = data['splits']['train']
    synth_qs = list(data.get('synth_qs') or [])
    n_synth = (len(synth_qs) if args.synth_per_real is None else
               min(len(synth_qs), int(len(real_qs) * args.synth_per_real + 0.5)))

    def loss_fn(logits_, quality_, batch_, slots_=None):
        """主损失与属性辅助损失共用同一套 --loss 配置（否则辅助项会反向拉扯操作点）。"""
        return compute_loss(logits_, quality_, batch_, device,
                            loss_kind=args.loss, pw_max=args.pw_max,
                            tv_alpha=args.tv_alpha, tv_beta=args.tv_beta,
                            slot_targets=slots_, alt_neg_w=args.alt_neg_w)
    print(f'每轮训练池：真实 {len(real_qs)} + 合成 {n_synth}/{len(synth_qs)}'
          '（社区隐藏前）')
    split_audit(data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    best_f1, best_state, bad = -1.0, None, 0
    _ep_secs = []
    for epoch in range(args.epochs):
        _t_ep = time.time()
        model.train()
        batch_qs = select_epoch_queries(real_qs, synth_qs,
                                        args.synth_per_real, sample_rng)
        rng.shuffle(batch_qs)
        losses = []
        used_real = used_synth = 0
        prompt_forms = defaultdict(int)
        neg_cov = [0, 0]                 # [带负点的提示数, 提示总数]（阶段16 的执行确认）
        alt_cov = [0, 0]                 # [B\\A 加压生效的查询数, 查询数]（阶段24 的执行确认）
        anon_cov = [0, 0]                # [被匿名种子的查询数, 查询数]（阶段27 的执行确认）
        for i in range(0, len(batch_qs), args.batch):
            batch = batch_qs[i:i + args.batch]

            # ---- 社区隐藏 episode：模拟测试设定（部分社区本 batch 零监督）----
            # 隐藏真实社区时，其扰动合成变体同步隐藏（堵住间接记忆通道）；
            # 候选只含真实训练社区：属性聚类圈的 based_on 是空列表，
            # 不能用它区分真实/合成，必须看 synth 标记。
            if args.hide_ratio > 0:
                batch_cids = list({int(q['target']['community_id']) for q in batch
                                   if not q.get('synth')})
                n_hide = int(len(batch_cids) * args.hide_ratio)
                if 0 < n_hide < len(batch_cids):
                    hidden = set(rng.sample(batch_cids, n_hide))

                    def leak(q):
                        if int(q['target']['community_id']) in hidden:
                            return True
                        bo = q.get('based_on')
                        return bool(bo) and any(c in hidden for c in bo)

                    batch = [q for q in batch if not leak(q)]
            if not batch:
                continue
            used_synth += sum(bool(q.get('synth')) for q in batch)
            used_real += sum(not q.get('synth') for q in batch)

            # ---- 节点特征 dropout：破坏节点 ID 记忆，逼模型用属性/结构模式 ----
            if args.feat_drop > 0:
                keep = (torch.rand_like(feat) >= args.feat_drop).float()
                feat_t = feat * keep
            else:
                feat_t = feat

            # ---- 种子点匿名（阶段27 路 A，--anon-seed-p）：**只让编码器看匿名图** ----
            # 关键：匿名张量只喂 `model.encoder`；下面 `decode(...)` 收到的仍是 `feat_t`
            #（提示 token = node_proj(feat[id])）⇒ 提示信息完整保留，只堵"编码器查节点身份"。
            # ⚠ --prompt-src H 下这条不成立（提示 token 就是编码器输出）⇒ main 里 ap.error 互斥。
            # 默认 0 ⇒ 不抽随机数、`feat_anon` 不构造 ⇒ 逐位等价旧行为。
            _anon_rows, _anon_q = anon_seed_rows(batch, args.anon_seed_p, rng, feat.size(0))
            if _anon_rows is None:
                H = model.encode(feat_t, edge_index)     # 编码一次（所有轮解码共享）
            else:
                feat_anon = feat_t.masked_fill(_anon_rows.to(feat_t.device)[:, None], 0.0)
                H = model.encode(feat_anon, edge_index)  # 编码一次（所有轮解码共享）
                anon_cov[0] += _anon_q
                anon_cov[1] += len(batch)
            # ---- 第 1 轮输入：显式提示形态混合或旧版 token dropout ----
            # 完整查询、仅正提示、单种子（可带属性）、随机子集各占约 1/4；
            # `--attr-only-p`（阶段23）> 0 时按该概率把某条提示换成"纯属性"形态（无节点 token）。
            sampled = [sample_train_prompt(q, args.prompt_mode, args.p_drop,
                                           prompt_rng if args.prompt_mode == 'mix' else rng,
                                           args.neg_keep, args.attr_only_p,
                                           args.prompt_cats == 'taxonomy',
                                           args.neg_cover)
                       for q in batch]
            plists = [plist for plist, _ in sampled]
            for _, form in sampled:
                prompt_forms[form] += 1
            # 负点覆盖读数（阶段16/17）：数**实际送进解码器**的提示。`--drop-neg-train`
            # 会把负提示整条剥掉（主分支 + 属性辅助支路），此时这里必须显示 0 ——
            # 首版数的是"采样态"（剥之前），在 drop_neg_train 下打出 ≈37%、与配置自相矛盾。
            _seen = ([strip_neg(pl) for pl in plists] if args.drop_neg_train else plists)
            neg_cov[0] += sum(1 for pl in _seen if any(p['pol'] == 1 for p in pl))
            neg_cov[1] += len(_seen)
            # B\\A 加压覆盖面（阶段24）：只在开关打开时统计（默认 0 ⇒ 不改任何输出/日志）。
            # 判据是"该查询真的有 B\\A"（候选数 ≥2 且减去 A 后非空），而不是"有没有 _alts"。
            if args.alt_neg_w > 0:
                _n_hat = 0
                for _q in batch:
                    _h = alt_neg_mask(_q, device)
                    if _h is not None and bool(_h.any()):
                        _n_hat += 1
                alt_cov[0] += _n_hat
                alt_cov[1] += len(batch)

            # ---- K 粒度候选的槽位顺序：每 batch 每查询随机打乱（防"槽位身份"捷径）----
            # --k-gran 1 时恒为 None → compute_loss 走旧路径，逐位等价。
            if args.k_gran > 1:
                slots = slot_targets_of(batch, args.k_gran, rng)
                slot_of = {id(q): slots[j] for j, q in enumerate(batch)}

                def aux_slots(qs_):
                    return [slot_of[id(q)] for q in qs_]
            else:
                slots = None

                def aux_slots(qs_):
                    return None

            # ---- 多轮交互：每轮预测 → 错误驱动补点 → 下一轮（末轮不补）----
            loss = 0.0
            for r in range(args.rounds):
                if args.drop_neg_train:
                    # 剥掉全部负 token（含交互轮补的 FP→负点）；整条提示都是负的时原样保留。
                    plists = [strip_neg(pl) or pl for pl in plists]
                logits, quality = decode(model, plists, feat_t, H, data, args)
                L = loss_fn(logits, quality, batch, slots)
                loss = loss + (1.0 if r == 0 else args.interact_weight) * L
                if r < args.rounds - 1:
                    nxt = []
                    for j, q in enumerate(batch):
                        # 补点用**部署口径**的掩码：K>1 时取质量头顶-1 的那个候选
                        k_pick = (int(torch.argmax(quality[j]))
                                  if args.k_gran > 1 else 0)
                        pred = (logits[j, k_pick].detach().sigmoid() > 0.5).cpu().numpy()
                        comm_set = set(torch.nonzero(q['_full']).reshape(-1).tolist())
                        nxt.append(plists[j] + error_driven_prompts(
                            comm_set, pred, rng, max_add=args.add_max))
                    plists = nxt

            # ---- 属性-only 辅助任务：直接训练"属性提示 → 圈掩码"映射 ----
            if args.attr_aux > 0:
                attr_plists, attr_batch = [], []
                for j, q in enumerate(batch):
                    pl = [p for p in make_prompt(q) if p['kind'] == 1]
                    # ⚠ 这条支路**也要同一道闸门**：上面那个 `kind == 1` 只过滤了类型、
                    #   没过滤极性，所以负属性（q['query']['neg_attrs']）会随正属性一起进来，
                    #   而本支路有自己的 decode、在轮循环之外，循环里那次剥点管不到它。
                    #   剥空则放弃该查询（不兜底为原提示——那等于把负属性又放回来）。
                    if args.drop_neg_train:
                        pl = strip_neg(pl)
                    if pl:
                        attr_plists.append(pl)
                        attr_batch.append(q)
                if attr_plists:
                    logits_a, quality_a = decode(model, attr_plists, feat_t, H,
                                                 data, args)
                    # 辅助任务与主损失用**同一份槽位顺序**（同一 epoch 同一条查询，
                    # 否则同一个查询在两处被训练到不同槽位，等于自相矛盾）
                    L_aux = loss_fn(logits_a, quality_a, attr_batch,
                                    aux_slots(attr_batch))
                    loss = loss + args.attr_aux * L_aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            if args.prior_lambda is not None:
                # 把 λ 钉在给定正值上（负锥只做减法，不给优化器改符号的机会）。
                # 每步覆盖而不是冻结参数：prior_w 是一个 (2,) Parameter，
                # 拆成两个 Parameter 会改动 state_dict 的键名，覆盖式更简单且不影响加载。
                with torch.no_grad():
                    model.decoder.prior_w[1] = args.prior_lambda
            losses.append(float(loss))

        H_val = model.encode(feat, edge_index)
        val_f1 = split_f1(model, H_val, data['splits']['val'], feat,
                          adj=data['adj'], shape=args.shape,
                          data=data, args=args)
        _ep_secs.append(time.time() - _t_ep)
        _avg_ep = (sum(_ep_secs[1:]) / len(_ep_secs[1:]) if len(_ep_secs) > 1
                   else _ep_secs[0])
        print(f'epoch {epoch:3d}  {_ep_secs[-1]:6.1f}s  平均 {_avg_ep:6.1f}s'
              f'  loss {np.mean(losses):.4f}  val F1 {val_f1:.4f}'
              f'  used real/synth {used_real}/{used_synth}'
              f'  prompt forms {dict(prompt_forms)}'
              f'  带负点 {neg_cov[0]}/{max(1, neg_cov[1])}'
              # 只有在 --alt-neg-w > 0 时才多打这一段：默认日志**逐字节不变**
              + (f'  B\\A加压 {alt_cov[0]}/{max(1, alt_cov[1])}'
                 if args.alt_neg_w > 0 else '')
              # 同理，只在 --anon-seed-p > 0 时多打（阶段27）
              + (f'  匿名 {anon_cov[0]}/{max(1, anon_cov[1])}'
                 if args.anon_seed_p > 0 else ''))
        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f'early stop @ {epoch}, best val F1 {best_f1:.4f}')
                break

        # ---- 轨迹诊断（每 --traj-every 轮 + 前 10 轮每轮；**只打印**，不参与任何选择）----
        # 动机（RUN-LIST 阶段21/22）：最终数字都取自 epoch 2~11 的快照，而 val 先冲高后回落。
        # output66 已经把成因定住了：峰值在 epoch 4 且 **test 也同意**（不是选点坏了），
        # 之后 val/test 一起下滑而**掩码均值从 51 震荡上行到 167（GT 53.6）**→ 过覆盖回潮。
        # ⚠ `epoch <= 10` 那段是**必须的**：峰值就在 epoch 4，而首版按 5 的倍数打，正好漏掉它。
        # 位置刻意放在**选择之后**：即使它出错，也不会影响已经做出的 best/bad 决定。
        # ⚠ 本块**不得**写 best_f1 / best_state / bad（守住在 temp/verify_traj.py，查 AST）。
    model.load_state_dict(best_state)                # 内存回载最优参数
    H_val = model.encode(feat, edge_index)
    groups = split_f1_groups(model, H_val, data['splits']['val'], feat, data['n2c'],
                             covered=covered_nodes(data, 'train'),
                             expo_of=community_exposure(data, 'val'),
                             adj=data['adj'], shape=args.shape,
                             data=data, args=args)
    print('val 分组 F1：' + '  '.join(
        f'{g} {v:.3f}（n={k}）' for g, (v, k) in groups.items()))
    return best_f1


# ==================== 评估 ====================
@torch.no_grad()
def evaluate(model, data, args, device):
    feat, edge_index, ei_np = data['feat'], data['edge_index'], data['ei_np']
    n2c, splits, n = data['n2c'], data['splits'], data['n']
    adj = data['adj']                      # 形状诊断用（邻接表）
    model.eval()
    H = model.encode(feat, edge_index)

    def decode1_all(plist):
        """单条提示的解码，返回**全部 K 个候选**：logits (K,n)、quality (K,)。

        只给 oracle / 多样性诊断用（best-of-K、候选两两 Jaccard）。
        """
        logits, quality = decode(model, [plist], feat, H, data, args)
        return logits[0], quality[0]

    def decode1(plist):
        """单条提示的解码，返回**部署口径的单个掩码**：(logits (n,), quality (K,))。

        选候选只走模块级的 `select_one`（与 `split_f1` / `split_f1_groups` 是**同一处**规则）；
        K=1 时它退化为唯一候选（`argmax` 返回 0）。
        ⚠ 返回的是 (n,) 而**不是** (K,n) —— 这样漏改调用点会**立刻报错**
        （对一维张量取 [0] 会得到标量、在 prf 里炸），而不是悄悄用错候选 0。
        「悄悄用错」比「报错」危险得多，所以这里刻意换了秩。
        """
        return select_one(*decode(model, [plist], feat, H, data, args))

    def group_name(q):
        seeds = q['query']['nodes'].tolist()
        return 'overlap' if any(len(n2c.get(int(v), [])) >= 2 for v in seeds) else 'plain'

    def mask_of(prob, plist):
        """概率向量 + **该处实际用的提示** -> 预测掩码。检索侧所有掩码都从这里出。

        ⚠ `plist` 是**必传**参数（不给就 TypeError）。理由：`--restrict-hop` 要靠它算种子球，
        漏传会**静默地不限制**，那是最坏的失败模式（日志看着正常，但规则其实没生效）。
        宁可让它在第一处调用就炸。

        顺序：**先按球限制、再走 `--shape`** —— 这样 LCC/trim 作用在受限掩码上，
        不会把球外的孤立点当成"最大连通分量"留下来。`--restrict-hop 0` 时球为 None，
        逐位等价于旧行为。
        """
        raw = prob > 0.5
        ball = prompt_ball(plist, data, args)
        if ball is not None:
            raw = np.logical_and(raw, ball)
        return shape_mask(raw, adj, args.shape)

    # ---- 检索精度 ----
    # ⚠ 分组表必须带"规模"列：P=R=F1=0.000 有三种完全不同的成因，只有规模能区分：
    #   ① 掩码为空（掩码均值≈0、空掩码=n）→ 是阈值/饱和问题；
    #   ② 掩码很大但整体错位（掩码均值大、空掩码=0）→ 是定位问题；
    #   ③ GT 本身为空（GT均值=0）→ 指标退化，任何模型都不可能得分。
    # plain 组在 30 份日志里 24 份恰好 0.000（含 --shape none 的运行），必须先把成因定下来。
    # ---- 提示类别评估表（阶段31）----
    # 8 个类别（正节点必备，utils.CAT_COMPS）+ 1 个随机对照 `rand` + 1 个历史诊断行 `attr`。
    # 全部诊断块共用这一份定义，保证"同名行在任何块里都是同一个提示"。
    # ⚠ `full` / `n+a` 与旧的 `+neg` / `base` 逐位相同（utils.make_prompt_cat 的守卫），
    #   所以历史日志的这两列可以直接和本表对照。
    # ⚠ `rand` = 正点+正属性+**同数量的随机**负点/负属性（同 pol、随机 id）：它把
    #   "负信息有用"和"只是多了两个 token"分开。按 seed 固定抽签、评测各块共用同一份缓存，
    #   所以同一次运行里所有诊断块看到的 rand 提示逐位相同。
    # ⚠ `attr`（纯属性、无节点 token）是历史诊断行，不参与早停/选择（guard: temp/verify_attr.py）。
    _rng_rand = random.Random(args.seed)
    _rand_cache = {id(q): make_prompt_rand(q, _rng_rand, data['n'], data['num_attrs'])
                   for q in splits['test']}
    EVAL_VARIANTS = ([(c, (lambda q, _c=c: make_prompt_cat(q, _c))) for c in PROMPT_CATS]
                     + [('rand', lambda q: _rand_cache[id(q)]),
                        ('attr', lambda q: make_prompt_attr(q))])

    print('\n== test 测试集（社区 token，阈值 0.5）==')
    print(f'{"variant":7s} {"group":10s} {"n":>4s} {"P":>6s} {"R":>6s} {"F1":>6s} '
          f'{"cond_p":>7s} {"cond_g":>7s} {"掩码均值":>8s} {"GT均值":>7s} {"空掩码":>6s}')
    prec_cat, rec_cat = {}, {}
    retrieval = {}
    per_q = {}        # vn -> {id(q): 逐查询 F1}；供下面的配对增益块复用，避免重复解码
    # 旧的 +neg / base 两列现在分别是 `full` / `n+a`；attr 行保留作历史对照。
    # `base − attr` 量"种子点贡献了多少"，`attr` 自己量"只有语义画像能到什么程度"。
    for vn, pfn in EVAL_VARIANTS:
        # 每条查询只解码一次，按组累加（10 个变体 × 3 组 ⇒ 省 3× 评估时间；读数与逐组重解码等价）
        acc = {g: {'ps': [], 'rs': [], 'f1s': [], 'cps': [], 'cgs': [],
                   'psz': [], 'gsz': [], 'nempty': 0}
               for g in ('all', 'overlap', 'plain')}
        for q in splits['test']:
            logits, _ = decode1(pfn(q))
            pred = mask_of(logits.sigmoid().cpu().numpy(), pfn(q))
            gt = q['_full'].cpu().numpy()
            p, r, f1 = prf(pred, gt)
            per_q.setdefault(vn, {})[id(q)] = f1
            _cp = conductance(ei_np, pred, n)
            _cg = conductance(ei_np, gt, n)
            _sz = int(pred.sum())
            _empty = int(_sz == 0)
            for grp in ('all', group_name(q)):
                d = acc[grp]
                d['ps'].append(p); d['rs'].append(r); d['f1s'].append(f1)
                d['cps'].append(_cp); d['cgs'].append(_cg)
                d['psz'].append(_sz); d['gsz'].append(int(gt.sum()))
                d['nempty'] += _empty
        for grp in ('all', 'overlap', 'plain'):
            d = acc[grp]
            if not d['f1s']:
                continue
            retrieval[(vn, grp)] = float(np.mean(d['f1s']))
            prec_cat[(vn, grp)] = float(np.mean(d['ps']))
            rec_cat[(vn, grp)] = float(np.mean(d['rs']))
            print(f'{vn:7s} {grp:10s} {len(d["f1s"]):4d} {np.mean(d["ps"]):6.3f} '
                  f'{np.mean(d["rs"]):6.3f} {np.mean(d["f1s"]):6.3f} '
                  f'{np.mean(d["cps"]):7.3f} {np.mean(d["cgs"]):7.3f} '
                  f'{np.mean(d["psz"]):8.1f} {np.mean(d["gsz"]):7.1f} {d["nempty"]:6d}')

    # ---- 阈值标定诊断（判断"调阈值"能带来多少；同一模型，不重训）----
    # 检索 F1 低的两种可能：① 概率场定位不准（排序本身就错）；② 排序可用但掩码
    # 规模没校准（阈值 0.5 硬编码，各 seed 的 R/P 比在 2.2~4.6 间跳）。
    # oracle-size = 每条查询取概率最高的 |GT| 个节点（不设阈值），给出①的上界：
    # 若它远高于 0.5 阈值的结果 → 是②，标定阈值值得做；若两者接近 → 是①，无用。
    # 阈值必须在 val 上选，故两侧都扫（只诊断，不改任何已报数字）。
    thr_grid = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def thr_scan(qs_):
        probs, gts = [], []
        for q in qs_:
            lg, _ = decode1(make_prompt(q))
            probs.append(lg.sigmoid().cpu().numpy())
            gts.append(q['_full'].cpu().numpy())
        f1_at = {}
        for t in thr_grid:
            f1_at[t] = float(np.mean([prf(p > t, g)[2] for p, g in zip(probs, gts)]))
        f1_or = []
        for p, g in zip(probs, gts):
            k = int(g.sum())
            pred = np.zeros_like(g)
            if k:
                pred[np.argpartition(-p, k - 1)[:k]] = True    # top-|GT| 个节点
            f1_or.append(prf(pred, g)[2])
        return f1_at, float(np.mean(f1_or))

    # ---- PLACE 可比口径：节点级 micro P/R/F1（摊平所有查询的节点预测）----
    # PLACE（CSAPeva.evalwithsplit）用 sklearn 在节点级算 micro-F1，与社区级
    # 最佳匹配 F1 不是同一个量：micro 被大社区主导，且不做集合匹配。
    # ---- 种子覆盖：种子点是否落在训练社区覆盖的节点上 ----
    covered = covered_nodes(data, 'train')
    comm_members = {int(c): set(map(int, m))
                    for c, m in data['g']['communities'].items()}
    # 社区 -> 划分归属（plain 专项诊断要用：最佳匹配落到 train 说明是记忆回退）
    cid_split = {int(q['target']['community_id']): s
                 for s in ('train', 'val', 'test')
                 for q in data['splits'][s]}
    train_cids = {int(q['target']['community_id']) for q in data['splits']['train']}
    test_cids = {int(q['target']['community_id']) for q in splits['test']}
    # ---- 成员重叠：目标社区成员与训练圈成员的重叠（≥0.3 视为存在近似孪生）----
    # 注意：社区级划分下测试社区的**标签/身份**从未参与训练（无泄露）；
    # 这里量的是**成员集合**的重叠——图本身社区高度重叠（1.40 社区/节点），
    # 与训练圈成员重合度高的测试社区可被"记忆的孪生"答对，不代表泛化。
    train_sets_exp = [comm_members[c] for c in train_cids if c in comm_members]
    expo_by_comm = {}
    for cid in test_cids:
        s = comm_members.get(cid, set())
        expo_by_comm[cid] = max((jaccard(s, t) for t in train_sets_exp),
                                default=0.0)
    # ---- 负提示消歧增益（legacy 口径：full − n+a，等价于旧的 +neg − base）----
    # 不再重复解码：直接复用检索表里逐查询的 F1（同一 checkpoint、同一批查询 ⇒ 配对）。
    # ---- 各提示类别相对 `n`（只有正节点）的配对增益（阶段31）----
    # 正节点是 8 个类别的公共底座 ⇒ 以它为基线，读的是"这一类额外加了什么"的净效果。
    # ---- K 粒度诊断（方案 4 的**一等读出**，必须先于任何指标看）----
    #   · 若 K 个候选几乎相同 → 多粒度等于没做（塌缩），后面指标不必看；
    #   · top-1（质量头选，= 部署口径）与 best-of-K（oracle）之差 = 选择器的好坏：
    #     两者接近 → 候选有了且会选；差距大 → 候选有了但选择器没学会；
    #   · 两者都是**同模型内配对**读出，不受运行间噪声影响 —— 正合"只追大效应"的判据。
    # ⚠ 只在**候选数 >1 的查询**上统计：候选只有 1 个（plain 组）时，多出来的槽位拿不到
    #   梯度（loss 里被 padding 屏蔽），它们的掩码是"没训过的"，计入会虚高 best-of-K。
    k_diag = None
    if args.k_gran > 1:
        _jac, _same, _f1_top, _f1_best, _hit, _n = [], [], [], [], [], 0
        for _split in ('val', 'test'):
            for q in data['splits'][_split]:
                if len(q.get('_alts') or []) < 2:
                    continue
                lg_all, q_all = decode1_all(make_prompt(q))
                masks = [mask_of(lg_all[k].sigmoid().cpu().numpy(), make_prompt(q))
                         for k in range(lg_all.size(0))]
                A_ = q['_full'].cpu().numpy()
                js = []
                for a_ in range(len(masks)):
                    for b_ in range(a_ + 1, len(masks)):
                        u = int(np.logical_or(masks[a_], masks[b_]).sum())
                        i_ = int(np.logical_and(masks[a_], masks[b_]).sum())
                        js.append(i_ / u if u else 1.0)
                _jac.append(float(np.mean(js)) if js else 1.0)
                _same.append(float(all(np.array_equal(masks[0], m)
                                       for m in masks[1:])))
                f1s = [prf(m, A_)[2] for m in masks]
                k_pick = int(torch.argmax(q_all))
                _f1_top.append(f1s[k_pick])
                _f1_best.append(max(f1s))
                _hit.append(float(f1s[k_pick] >= max(f1s) - 1e-12))
                _n += 1
        if _n:
            k_diag = dict(jac=float(np.mean(_jac)), same=float(np.mean(_same)),
                          top1=float(np.mean(_f1_top)), best=float(np.mean(_f1_best)),
                          hit=float(np.mean(_hit)), n=_n)
            print('\n== K 粒度诊断（方案 4 一等读出；仅统计候选数 >1 的查询）==')
            print(f'  n={_n}  K={args.k_gran}')
            print(f'  候选两两 Jaccard 均值 {k_diag["jac"]:.3f}'
                  f'（=1 → K 个候选完全相同，多粒度没生效）')
            print(f'  全 K 相同的查询比例 {k_diag["same"] * 100:.1f}%')
            print(f'  top-1（质量头选，部署口径）F1 {k_diag["top1"]:.3f}  vs  '
                  f'best-of-K（oracle）F1 {k_diag["best"]:.3f}  '
                  f'→ 选择器命中率 {k_diag["hit"] * 100:.1f}%')
            print('  判读：Jaccard 明显 <1 = 候选确有分歧（机制生效）；'
                  'top-1 ≈ best = 质量头会选；差距大 = 候选有了但选择器没学会')

    # ---- 时延微基准 ----
    # 诊断块已删除：给 return 里引用的名字填默认值
    _nan = float('nan')
    gains, expo_ret, comm_ret, micro_ret = {}, {}, {}, {}
    k_diag = None
    collapse_train = collapse_test = _nan
    interact_gain = _nan
    thr_oracle_test = thr_best_val = _nan
    thr_best_val_f1 = thr_best_val_test_f1 = thr_f1_test_05 = _nan
    thr_n_val = 0
    return dict(
        k_jac=(k_diag or {}).get('jac', 0.0),
        k_same=(k_diag or {}).get('same', 0.0),
        k_top1=(k_diag or {}).get('top1', 0.0),
        k_best=(k_diag or {}).get('best', 0.0),
        k_hit=(k_diag or {}).get('hit', 0.0),
        k_n=(k_diag or {}).get('n', 0.0),
        # 旧 +neg / base 两列 = 类别表的 full / n+a（逐位相同，历史可直接对照）
        f1_neg=retrieval.get(('full', 'all'), 0.0),
        p_neg=prec_cat.get(('full', 'all'), 0.0),
        r_neg=rec_cat.get(('full', 'all'), 0.0),
        p_base=prec_cat.get(('n+a', 'all'), 0.0),
        r_base=rec_cat.get(('n+a', 'all'), 0.0),
        **{'p_cat_%s' % c: prec_cat.get((c, 'all'), float('nan'))
           for c in list(PROMPT_CATS) + ['rand', 'attr']},
        **{'r_cat_%s' % c: rec_cat.get((c, 'all'), float('nan'))
           for c in list(PROMPT_CATS) + ['rand', 'attr']},

        f1_base=retrieval.get(('n+a', 'all'), 0.0),
        f1_neg_overlap=retrieval.get(('full', 'overlap'), 0.0),
        f1_base_overlap=retrieval.get(('n+a', 'overlap'), 0.0),
        f1_neg_plain=retrieval.get(('full', 'plain'), 0.0),
        f1_base_plain=retrieval.get(('n+a', 'plain'), 0.0),
        gain_all=gains.get('all', 0.0),
        gain_overlap=gains.get('overlap', 0.0),
        gain_plain=gains.get('plain', 0.0),
        micro_f1_base=micro_ret.get('n+a', 0.0),
        # ---- 阶段31：8 类别逐类读数（macro all/overlap/plain + micro + 相对 n 的配对增益）----
        **{'cat_all_%s' % c: retrieval.get((c, 'all'), 0.0) for c in PROMPT_CATS},
        **{'cat_ov_%s' % c: retrieval.get((c, 'overlap'), 0.0) for c in PROMPT_CATS},
        **{'cat_pl_%s' % c: retrieval.get((c, 'plain'), 0.0) for c in PROMPT_CATS},
        cat_all_rand=retrieval.get(('rand', 'all'), 0.0),
        cat_pl_rand=retrieval.get(('rand', 'plain'), 0.0),
        cat_all_attr=retrieval.get(('attr', 'all'), 0.0),
        interact_gain=interact_gain,
        collapse_train=collapse_train,
        collapse_test=collapse_test,
        enc_ms=float('nan'), dec_ms=float('nan'),
    )


# ==================== ACG 探测 ====================
def jaccard(a, b):
    return len(a & b) / len(a | b) if len(a | b) else 0.0


# ==================== 输出"形状"诊断用的掩码处理 ====================
def build_adj(edge_index, n):
    """邻接表（list of list）。边表是对称的（无向边存两次），所以只按 src 侧
    追加即可让每个邻居恰好出现一次。"""
    adj = [[] for _ in range(n)]
    for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[a].append(b)
    return adj


def trim_isolated(mask, adj):
    """去掉掩码内"没有掩码内邻居"的节点（1 跳裁剪）。

    比取最大连通分量温和得多：只削掉弥散的孤立边缘，主体不动。
    用来判断"预测集合里那些散落节点是不是纯噪声"。
    """
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return mask
    S = set(idx.tolist())
    out = np.zeros_like(mask)
    for v in idx.tolist():
        if any(u in S for u in adj[v]):
            out[v] = True
    return out


def shape_mask(mask, adj, shape='lcc'):
    """按 --shape 把预测掩码整理成"社区形状"——**所有输出掩码的唯一整理入口**。

    none = 原样（阈值 0.5 直接输出）
    trim = 去孤立边缘（削掉掩码内没有掩码内邻居的节点）
    lcc  = 最大连通分量（默认）

    依据：本数据集的真实社区本身有 95% 是一个连通块（日志里
    "最大连通分量占比 中位 1.00 vs 训练社区 0.95"），所以"输出取连通分量"
    与目标形态匹配；实测（output35）节点+属性侧 AvgF1 0.240 → 0.277（配对比较）。
    """
    if shape == 'lcc':
        return largest_component(mask, adj)
    if shape == 'trim':
        return trim_isolated(mask, adj)
    return mask


def largest_component(mask, adj):
    """掩码诱导子图的最大连通分量（等价于 CLARE 那种"连通 ego-net"的形状假设）。"""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return mask
    S = set(idx.tolist())
    seen, best = set(), []
    for start in idx.tolist():
        if start in seen:
            continue
        comp, stack = [start], [start]
        seen.add(start)
        while stack:
            v = stack.pop()
            for u in adj[v]:
                if u in S and u not in seen:
                    seen.add(u)
                    comp.append(u)
                    stack.append(u)
        if len(comp) > len(best):
            best = comp
    out = np.zeros_like(mask)
    out[best] = True
    return out


def dedup_cands(cands_f, tau_j, key='quality'):
    """按 key 降序做 Jaccard 贪心去重，返回保留集。

    保留原候选的全部字段（mask/quality/stability/...），只补 nodeset——
    后续的过滤/统计会继续用到这些字段。
    （曾试验过"包含式去重"以去掉被包含的小社区，因 facebook 真实圈本身存在
    嵌套——min_size≥16 时 6 个社区被完全包含——开启会伤 found，已放弃。）
    """
    kept = []
    seen = {}
    for c in sorted(cands_f, key=lambda c: -c[key]):
        s = set(np.nonzero(c['mask'])[0].tolist())
        if tau_j >= 1:
            # 仅合并完全相同的输出 → 用哈希 O(k)；全节点种子（--M 0）时两两比较是 O(k·|kept|)
            fs = frozenset(s)
            if fs in seen:
                continue
            seen[fs] = True
        elif any(jaccard(s, k.get('nodeset', ())) >= tau_j for k in kept):
            continue
        kept.append(dict(c, nodeset=s))
    if kept and cands_f:
        assert set(kept[0]) == set(cands_f[0]) | {'nodeset'}, \
            'dedup_cands 丢失字段（下游会 KeyError，如按 stability 过滤）'
    return kept


def onmi_score(pred_comms, true_comms):
    """ONMI（McDaid 2011）：1 - 0.5*(H(X|Y) + H(Y|X))。

    移植 CLARE utils/metrics.py 的 get_nmi_score（ProCom 同源），其中
    H(x|y) 取对 y 的最小条件熵；额外加了 H(x)=0 的保护。
    """
    from math import log
    if not pred_comms or not true_comms:
        return 0.0
    pred_comms = [set(map(int, c)) for c in pred_comms]
    true_comms = [set(map(int, c)) for c in true_comms]
    tot = len({n for c in pred_comms + true_comms for n in c})
    if tot == 0:
        return 0.0

    def h(x):
        return -x * (log(x) / log(2)) if x > 0 else 0.0

    def H(c):
        p1 = len(c) / tot
        return h(1 - p1) + h(p1)

    def h_joint(xi, yj):
        inter = len(xi & yj)
        p11, p10, p01 = inter / tot, (len(xi) - inter) / tot, (len(yj) - inter) / tot
        p00 = 1 - p11 - p10 - p01
        if h(p11) + h(p00) >= h(p01) + h(p10):
            return h(p11) + h(p10) + h(p01) + h(p00)
        return H(xi) + H(yj)

    def H_given(X, Y):
        s = 0.0
        for xi in X:
            hx = H(xi)
            if hx <= 0:
                continue
            s += min(h_joint(xi, yj) - H(yj) for yj in Y) / hx
        return s / len(X)

    return 1 - 0.5 * (H_given(pred_comms, true_comms) + H_given(true_comms, pred_comms))


def partition_seeds(edge_index, n, M, rng, rounds=8, min_cell=5):
    """按结构划分挑种子：标签传播 → 合并过小 cell → max-min 分配（每 cell 至少 1 个）。

    动机是随机采样可能整片漏掉小社区（M=300 时平均 ~5.8/56 个社区零种子）。
    但**实测在同一预算下覆盖反而更差**（真实 circles，见 RUN-LIST 阶段 2.5）：
    保证性条件是"GT 社区完整包含某个 cell"，只对 21~22/56 成立；每 cell 一个
    种子还会把预算花在 GT 并不需要的碎片上、又欠采大 blob（一个 17 节点社区
    躲在 176 节点 cell 里，只拿到 1 个位于 hub 的种子 → 恰好落在社区外）。
    故默认仍为均匀随机；本函数仅作消融对照。返回 (seeds, info)。
    """
    src = np.asarray(edge_index[0]).astype(np.int64)
    dst = np.asarray(edge_index[1]).astype(np.int64)
    adj = [[] for _ in range(n)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)
        adj[v].append(u)
    deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)

    labels = np.arange(n)
    for _ in range(rounds):
        moved = 0
        for v in rng.permutation(n).tolist():
            nb = adj[v]
            if not nb:
                continue
            arr = labels[np.fromiter(nb, dtype=np.int64, count=len(nb))]
            best = int(np.argmax(np.bincount(arr)))      # 平票取最小标签 → 确定性
            if labels[v] != best:
                labels[v] = best
                moved += 1
        if moved == 0:
            break

    cells = defaultdict(list)
    for v in range(n):
        cells[int(labels[v])].append(v)
    cells = [set(c) for c in cells.values()]

    for _ in range(4):                                   # 合并碎片（否则预算浪费在 1~2 节点 cell）
        node2cell = {}
        for i, c in enumerate(cells):
            for v in c:
                node2cell[v] = i
        link = defaultdict(Counter)
        for v in range(n):
            i = node2cell.get(v)
            if i is None:
                continue
            for u in adj[v]:
                j = node2cell.get(u)
                if j is not None and j != i:
                    link[i][j] += 1
        small = [i for i, c in enumerate(cells) if len(c) < min_cell and link[i]]
        if not small:
            break
        drop = set()
        for i in small:
            j = link[i].most_common(1)[0][0]
            if j in drop:
                continue
            cells[j] |= cells[i]
            drop.add(i)
        cells = [c for i, c in enumerate(cells) if i not in drop]

    order_in_cell = [sorted(c, key=lambda v: -deg[v]) for c in cells]
    seeds, used = [], set()
    for cl in order_in_cell:                             # 第一轮：每 cell 一个（度最大）
        if len(seeds) >= M:
            break
        seeds.append(cl[0])
        used.add(cl[0])
    budget = M - len(seeds)
    idx = [1] * len(order_in_cell)
    pool = sorted(range(len(cells)), key=lambda i: -len(cells[i]))
    while budget > 0:                                    # 余量：按 cell 规模轮转补种
        progressed = False
        for i in pool:
            if budget <= 0:
                break
            cl = order_in_cell[i]
            while idx[i] < len(cl) and cl[idx[i]] in used:
                idx[i] += 1
            if idx[i] < len(cl):
                seeds.append(cl[idx[i]])
                used.add(cl[idx[i]])
                idx[i] += 1
                budget -= 1
                progressed = True
        if not progressed:
            break
    info = dict(K=len(cells), sizes=sorted(len(c) for c in cells))
    return np.asarray(sorted(set(seeds)), dtype=np.int64)[:M], info


@torch.no_grad()
def acg_eval(model, data, args, device):
    feat, edge_index = data['feat'], data['edge_index']
    n = data['n']
    adj = data['adj']                      # 形状诊断用（邻接表）
    communities = data['g']['communities']
    test_cids = {int(q['target']['community_id']) for q in data['splits']['test']}
    train_cids = {int(q['target']['community_id']) for q in data['splits']['train']}
    model.eval()
    H = model.encode(feat, edge_index)

    node2first = -np.ones(n, dtype=np.int64)         # 主归属（重叠节点取第一个）
    for cid, members in communities.items():
        for v in members:
            if node2first[int(v)] < 0:
                node2first[int(v)] = int(cid)

    if args.M <= 0 or args.M >= n:
        seeds = np.arange(n)                         # 全节点做种子：消除"社区没被采到"的覆盖瓶颈
    elif args.acg_seed == 'partition':
        seeds, pinfo = partition_seeds(edge_index, n, args.M,
                                       np.random.RandomState(args.seed))
        _ps = pinfo['sizes']
        print(f'== ACG 种子=结构划分 ==  cell {pinfo["K"]} 个；规模 min/中位/max = '
              f'{_ps[0]}/{_ps[len(_ps) // 2]}/{_ps[-1]}；每 cell ≥1 个后按规模补到 {len(seeds)}')
    else:
        seeds = np.random.choice(n, size=args.M, replace=False)
    if data.get('attr_ptr') is not None:
        _ptr, _idx = data['attr_ptr'], data['attr_idx']
        top_attrs = {v: _idx[_ptr[v]:_ptr[v + 1]].tolist() for v in range(n)}  # 种子的属性提示
    else:
        X_np = np.asarray(data['g']['attributes'])
        top_attrs = {v: np.nonzero(X_np[v])[0].tolist() for v in range(n)}  # 种子的属性提示

    # 种子覆盖诊断：测试社区里有多少个连一个种子都没有（这些社区按构造不可能被检出）
    seed_set = set(int(s) for s in seeds)
    seed_cov = {int(c): len(set(map(int, communities[c])) & seed_set) for c in test_cids}
    zero_seed = [c for c, k in seed_cov.items() if k == 0]
    _cv = list(seed_cov.values()) or [0]
    print(f'== ACG 种子覆盖 ==  种子数 {len(seeds)}/{n}；测试社区 {len(test_cids)} 个中'
          f' 零种子 {len(zero_seed)} 个；社区内种子数 min/中位/max = '
          f'{min(_cv)}/{int(np.median(_cv))}/{max(_cv)}')
    if zero_seed:
        print(f'  ⚠ 零种子社区（构造上不可检出）：{sorted(zero_seed)}')

    def run_acg(attr_k):
        """跑一遍 ACG 管线（attr_k = 每种子附带的属性提示数），返回 (kept, raw_cands, small_cnt)。
        attr_k=0 为节点-only 形态；attr_k>0 时附带种子的 top 属性（AFN 式查询）。"""
        cands, small = [], 0
        # K>1 时统计"若把 K 个候选全发"会多出多少条（只报数、不计入任何指标），
        # 用来佐证"我们刻意只发 top-1，以避免输出规模膨胀稀释对称 AvgF1"
        extra_cnt = 0
        B = 64
        for i in range(0, len(seeds), B):
            chunk = seeds[i:i + B]
            plists = []
            for s in chunk:
                pl = [{'kind': 0, 'id': int(s), 'pol': 0}]
                if attr_k > 0:
                    for a in top_attrs.get(int(s), [])[:attr_k]:
                        pl.append({'kind': 1, 'id': int(a), 'pol': 0})
                plists.append(pl)
            logits, quality = decode(model, plists, feat, H, data, args)
            prob_all = logits.sigmoid().cpu().numpy()      # (B,K,n)
            q_all = quality.detach().cpu().numpy()         # (B,K)
            # 方案 4：每个种子只发**质量头顶-1** 的那个候选 —— 保持输出条数与 K=1 可比
            # （把 K 个都吐出去会让输出数 ×K，按阶段4 的结论会稀释对称 AvgF1 的 pred 侧，
            #  那样即使候选更好也会被指标惩罚）。K=1 时 argmax 恒为 0，逐位等价旧行为。
            if args.k_gran > 1:
                extra_cnt += int(args.k_gran - 1) * len(chunk)
            pick = (q_all.argmax(axis=1) if args.k_gran > 1
                    else np.zeros(len(chunk), dtype=np.int64))
            for j in range(len(chunk)):
                prob = prob_all[j, int(pick[j])]
                mask_raw = prob > 0.5                    # 形状处理前（形状诊断要用它）
                # `--restrict-hop`：ACG 的提示正点就是这个种子，所以球就是种子的 k 跳邻域
                _ball = prompt_ball(plists[j], data, args)
                if _ball is not None:
                    mask_raw = np.logical_and(mask_raw, _ball)
                mask = shape_mask(mask_raw, adj, args.shape)
                if mask.sum() < 3:
                    small += 1
                    continue
                lgt = np.log(prob + 1e-9) - np.log(1 - prob + 1e-9)
                cands.append(dict(mask=mask, mask_raw=mask_raw,
                                  quality=float(q_all[j, int(pick[j])]),
                                  stability=stability_score(lgt)))
        raw_cands = list(cands)                      # 过滤前副本（未过滤诊断/标定用）
        return (raw_cands, raw_cands, small_cnt, len(raw_cands), 0)

    results_acg = {}
    small_cnt = 0
    for mode_name, attr_k in (('节点-only', 0), ('节点+属性', args.acg_attr)):
        _t0 = time.time()
        _t_mode = time.time()
        kept, raw_cands, small_cnt, n_filtered, extra_cnt = run_acg(attr_k)
        _dt = time.time() - _t0
        print(f'\n== ACG 探测[{mode_name}]（全图生成，F1 仅评测试社区）==')
        print(f'过滤后 {n_filtered} -> 去重后 {len(kept)}；'
              f'种子输出塌缩为空 {small_cnt}/{len(seeds)}；耗时 {_dt:.1f}s（{len(seeds)} 种子）')
        if args.k_gran > 1:
            print(f'  [K 粒度] 只发质量头顶-1（与 K=1 的输出条数可比）；'
                  f'若把 K={args.k_gran} 个候选全发会多出约 {extra_cnt} 条（未计入指标）')

        gt_test = {}
        for cid, members in communities.items():
            if int(cid) not in test_cids:
                continue
            gt = np.zeros(n, dtype=bool)
            gt[[int(v) for v in members]] = True
            gt_test[int(cid)] = gt

        # 成员重叠分组：测试社区成员与训练社区成员的节点级重叠（≥0.3 视为存在近似孪生）
        train_sets = [set(map(int, m)) for c, m in communities.items()
                      if int(c) in train_cids]
        def exposure(cid):
            s = set(map(int, communities[cid]))
            return max((jaccard(s, t) for t in train_sets), default=0.0)
        expo_hi = [c for c in gt_test if exposure(c) >= 0.3]
        expo_lo = [c for c in gt_test if exposure(c) < 0.3]

        def expo_stats(cids_):
            if not cids_:
                return 0.0, 0, 0
            f1s = [max((prf(k['mask'], gt_test[c])[2] for k in kept), default=0.0)
                   for c in cids_]
            return float(np.mean(f1s)), int(sum(f > 0.5 for f in f1s)), len(f1s)

        def expo_stats_raw(cids_):
            """未过滤口径：不依赖 tau，kept 为空时仍可读。"""
            if not cids_:
                return 0.0, 0, 0
            f1s = [max((prf(cand['mask'], gt_test[cid])[2] for cand in raw_cands),
                       default=0.0) for cid in cids_]
            return float(np.mean(f1s)), int(sum(f > 0.5 for f in f1s)), len(f1s)

        # 种子覆盖分组：把"有种子却没检出"与"根本没种子（构造上不可检出）"分开
        cov_hi = [c for c in gt_test if seed_cov.get(c, 0) > 0]
        cov_zero = [c for c in gt_test if seed_cov.get(c, 0) == 0]
        chi_f1, chi_found, chi_n = expo_stats_raw(cov_hi)
        cze_f1, cze_found, cze_n = expo_stats_raw(cov_zero)
        print(f'  种子覆盖分组[未过滤]：有种子 n={chi_n} best-F1 {chi_f1:.3f} '
              f'found {chi_found}/{chi_n}；零种子 n={cze_n} best-F1 {cze_f1:.3f} '
              f'found {cze_found}/{cze_n}')

        # 真值侧：每个测试社区在输出集合中的最佳匹配 F1
        f1_true = []
        for cid, gt in gt_test.items():
            best = 0.0
            for k in kept:
                best = max(best, prf(k['mask'], gt)[2])
            f1_true.append(best)

        # 预测侧：每个输出社区只匹配测试社区（其他社区的输出也计入均值）
        f1_pred = []
        for k in kept:
            best = 0.0
            for gt in gt_test.values():
                best = max(best, prf(k['mask'], gt)[2])
            f1_pred.append(best)
        node2pred = -np.ones(n, dtype=np.int64)
        for pi, k in enumerate(kept):
            for v in k.get('nodeset', ()):
                if node2pred[v] < 0:
                    node2pred[v] = pi
        covered = node2pred >= 0
        nmi = normalized_mutual_info_score(node2first[covered], node2pred[covered]) \
            if covered.any() else 0.0
        ari = adjusted_rand_score(node2first[covered], node2pred[covered]) \
            if covered.any() else 0.0
        sym_f1 = 0.5 * (float(np.mean(f1_true)) + float(np.mean(f1_pred))) \
            if f1_true and f1_pred else 0.0
        print(f'[过滤+去重后] 测试社区 {len(gt_test)}  '
              f'真值侧 best-F1 {np.mean(f1_true) if f1_true else 0.0:.3f}  '
              f'预测侧 best-F1 {np.mean(f1_pred) if f1_pred else 0.0:.3f}  '
              f'对称均值 {sym_f1:.3f}  '
              f'>0.5 的测试社区 {sum(f > 0.5 for f in f1_true)}/{len(f1_true)}')
        print('  NMI/ARI/覆盖率仍按全图覆盖节点计算')

        # ---- CLARE / ProCom 同口径（对称 best-F1 / Jaccard / ONMI / detect% + 配对）----
        # 两侧各自取最佳匹配再平均，不做一一配对；pairwise 为 Dice 与 Jaccard。
        # CLARE/ProCom 直接输出全部预测社区（无过滤步骤），故**未过滤集合**才是可比
        # 口径；过滤+去重后的集合是本方法的实际输出，作为后处理消融并列报告。
        gt_sets = {cid: set(np.nonzero(gt)[0].tolist()) for cid, gt in gt_test.items()}
        gt_nodes = set().union(*gt_sets.values()) if gt_sets else set()

        def clare_block(masks, nsets, label):
            # F1 用 bool 掩码（gt_test），Jaccard 用集合（gt_sets）——两者别混
            f1t = [max((prf(m, gt)[2] for m in masks), default=0.0)
                   for gt in gt_test.values()]
            f1p = [max((prf(m, gt)[2] for gt in gt_test.values()), default=0.0)
                   for m in masks]
            sf1 = (0.5 * (float(np.mean(f1t)) + float(np.mean(f1p)))
                   if f1t and f1p else 0.0)
            jt = [max((jaccard(s, g) for s in nsets), default=0.0)
                  for g in gt_sets.values()]
            jp = [max((jaccard(s, g) for g in gt_sets.values()), default=0.0)
                  for s in nsets]
            sj = (0.5 * (float(np.mean(jt)) + float(np.mean(jp)))
                  if jt and jp else 0.0)
            om = onmi_score(nsets, list(gt_sets.values()))
            pn = set().union(*nsets) if nsets else set()
            dp = len(gt_nodes & pn) / len(gt_nodes) if gt_nodes else 0.0
            prs = sorted(((prf(masks[i], gt_test[cid])[2], cid, i)
                          for cid in gt_test for i in range(len(masks))),
                         key=lambda t: -t[0])
            uc, up = set(), set()
            for f1v, cid, i in prs:
                if f1v < 0.5 or cid in uc or i in up:
                    continue
                uc.add(cid)
                up.add(i)
            d_p = len(up) / len(masks) if masks else 0.0
            d_r = len(uc) / len(gt_sets) if gt_sets else 0.0
            d_f1 = 2 * d_p * d_r / (d_p + d_r) if d_p + d_r > 0 else 0.0
            hi_best = (float(np.mean([max((prf(m, gt_test[c])[2] for m in masks),
                                          default=0.0) for c in expo_hi]))
                       if expo_hi else 0.0)
            lo_best = (float(np.mean([max((prf(m, gt_test[c])[2] for m in masks),
                                          default=0.0) for c in expo_lo]))
                       if expo_lo else 0.0)
            print(f'  CLARE 口径[{label}]：n={len(masks)}  AvgF1 {sf1:.3f}  '
                  f'AvgJaccard {sj:.3f}  ONMI {om:.3f}  detect% {dp:.3f}  found '
                  f'{sum(f > 0.5 for f in f1t)}/{len(f1t)}  配对 F1 {d_f1:.3f}  '
                  f'重叠≥0.3 best-F1 {hi_best:.3f}  低重叠 best-F1 {lo_best:.3f}')
            return dict(sym_f1=sf1, sym_j=sj, onmi=om, detect_pct=dp,
                        found=int(sum(f > 0.5 for f in f1t)),
                        det_p=d_p, det_r=d_r, det_f1=d_f1, det_matched=len(uc),
                        matched_hi=sum(c in uc for c in expo_hi),
                        matched_lo=sum(c in uc for c in expo_lo))

        clare_raw = clare_block(
            [c['mask'] for c in raw_cands],
            [set(np.nonzero(c['mask'])[0].tolist()) for c in raw_cands],
            '未过滤=全部种子输出，与 CLARE/ProCom 可比（主指标）')
        # 只保留 CLARE 未过滤口径：其余诊断（过滤去重/NMI/ARI/后处理对照/等池/形状/阈值扫描）一律不算
        _res = {k: clare_raw.get(k, float('nan')) for k in (
            'sym_f1', 'sym_j', 'onmi', 'detect_pct', 'found', 'det_f1')}
        # 重叠分组的两条 best-F1 来自 expo_stats_raw（不在 clare_raw 里）
        _res['expo_hi_f1_raw'] = expo_stats_raw(expo_hi)[0]
        _res['expo_lo_f1_raw'] = expo_stats_raw(expo_lo)[0]
        print(f'[time] ACG[{mode_name}] {time.time() - _t_mode:.1f}s', flush=True)
        results_acg[mode_name] = _res
        continue
        clare_kept = clare_block([k['mask'] for k in kept],
                                 [k.get('nodeset', ()) for k in kept],
                                 '过滤+去重后=本方法实际输出（后处理消融）')
        print(f'  配对命中（成员重叠≥0.3 / <0.3）[未过滤]：{clare_raw["matched_hi"]}/'
              f'{len(expo_hi)}  vs  {clare_raw["matched_lo"]}/{len(expo_lo)}')

        # ---- 后处理规则对照（复用同一批候选，不重训）----
        # 回答"过滤/去重是否必要、以什么顺序/排序做"：全部候选 → 仅过滤 → 两种去重排序
        # → 先去重再过滤（让代表选择不依赖 quality 头）→ 宽阈值（IoU 0.33 ≈ Dice 0.5）。
        filt = [c for c in raw_cands
                if c['quality'] >= args.tau_q and c['stability'] >= args.tau_s]
        print('  == 后处理规则对照（复用候选，不重训）==')
        for label, cs in (
                ('仅过滤（不去重）', filt),
                (f'过滤→{args.dedup_key} 去重（当前配置）', kept),
                ('过滤→stability 去重', dedup_cands(filt, args.tau_j, key='stability')),
                ('stability 去重→再过滤',
                 [c for c in dedup_cands(raw_cands, args.tau_j, key='stability')
                  if c['quality'] >= args.tau_q and c['stability'] >= args.tau_s]),
                ('宽阈值 q≥0.33→stability 去重',
                 dedup_cands([c for c in raw_cands
                              if c['quality'] >= 0.33 and c['stability'] >= args.tau_s],
                             args.tau_j, key='stability')),
        ):
            clare_block([c['mask'] for c in cs],
                        [c['nodeset'] if 'nodeset' in c
                         else set(np.nonzero(c['mask'])[0].tolist()) for c in cs],
                        label)
        # ---- 等池对照（排掉"池子更分散 → 最大值自己抬高"这一替代解释）----
        # AvgF1 / 重叠 best-F1 都是"在候选池里**取最大**"，所以池子内容更分散也可能把最大值
        # 抬起来，而每个掩码其实没变好。把两个配置的池子截到**同一条数**再算，就控制住了这一点。
        # 两行分工：
        #   · 随机 N（固定种子）→ 纯等量对照；跨配置比较只看这一行。
        #   · quality 前 N → 顺带检验质量头的**排序**有没有用（它 p10~p90 只有 0.23~0.29，
        #     若"前 N"和"随机 N"一样，说明排序≈随机，选择器是摆设）。
        # 判读：开 pos 后随机 N 行仍高于对照的同名行 → 是真变好；只有全池涨、等池不涨 →
        # 是抽奖效应，检测侧那个增益就不能算。
        _EQ_N = 2000
        _n_eq = min(_EQ_N, len(raw_cands))
        if _n_eq >= 8:
            _rs = np.random.RandomState(0)          # 自成发生器：不碰全局随机流
            _idx = _rs.choice(len(raw_cands), _n_eq, replace=False)
            _sub = [raw_cands[i] for i in _idx]
            clare_block([c['mask'] for c in _sub],
                        [set(np.nonzero(c['mask'])[0].tolist()) for c in _sub],
                        f'等池对照：随机 {_n_eq}（种子 0）')
            _subq = sorted(raw_cands, key=lambda c: -c['quality'])[:_n_eq]
            clare_block([c['mask'] for c in _subq],
                        [set(np.nonzero(c['mask'])[0].tolist()) for c in _subq],
                        f'等池对照：quality 前 {_n_eq}')
            print(f'  判读：等池两行与对照的同名行比（条数取 min({_EQ_N}, 本池)）——'
                  f'随机 {_n_eq} 仍更高 = 真变好；'
                  f'若"quality 前 {_n_eq}"明显高于"随机 {_n_eq}"则质量头排序有用，'
                  f'否则选择器≈随机（这也是搜索侧 top-1 选不准的原因）。')
        # ---- 形状诊断（检测侧）：把候选掩码做成"更像社区"的形状后重算 CLARE 口径 ----
        # 与 evaluate() 里检索侧的形状诊断同源：判断"输出集合弥散"是不是真值侧偏低的
        # 原因。GT 本身偏散（小社区 conductance 0.756），所以连通不一定更好，故并列两档。
        # **必须用 mask_raw**：候选生成时已按 --shape 处理过，再处理一次是幂等的，
        # 那样三行会完全相同、诊断失效（--shape lcc 时踩过）。
        for _shape, _fn in (('去孤立边缘', trim_isolated),
                            ('最大连通分量', largest_component)):
            _ms = [_fn(c.get('mask_raw', c['mask']), adj) for c in raw_cands]
            _sets = [set(np.nonzero(m)[0].tolist()) for m in _ms]
            clare_block(_ms, _sets, f'形状诊断：未过滤+{_shape}')
        # 形状诊断的"原样"行 = 未经形状处理的原始掩码，作为上面两行的基线
        _mr = [c.get('mask_raw', c['mask']) for c in raw_cands]
        clare_block(_mr, [set(np.nonzero(m)[0].tolist()) for m in _mr],
                    '形状诊断：未过滤+原样（raw 基线）')
        hi_f1, hi_found, hi_n = expo_stats(expo_hi)
        lo_f1, lo_found, lo_n = expo_stats(expo_lo)
        print(f'  成员重叠组（与训练圈成员 max Jaccard ≥0.3）：n={hi_n}  '
              f'best-F1 {hi_f1:.3f}  found {hi_found}/{hi_n}')
        print(f'  低重叠组（<0.3）：n={lo_n}  best-F1 {lo_f1:.3f}  found {lo_found}/{lo_n}')
        rhi_f1, rhi_found, rhi_n = expo_stats_raw(expo_hi)
        rlo_f1, rlo_found, rlo_n = expo_stats_raw(expo_lo)
        print(f'  未过滤口径：重叠组 found {rhi_found}/{rhi_n}（best-F1 {rhi_f1:.3f}）  '
              f'低重叠组 found {rlo_found}/{rlo_n}（best-F1 {rlo_f1:.3f}）')

        if not kept:
            # 未过滤诊断：过滤前的全部种子掩码与测试社区的最佳匹配
            print(f'  [未过滤诊断] {len(seeds)} 种子中单点输出 ≥3 节点的：{len(raw_cands)}'
                  f'（塌缩为空 {small_cnt}）')
            f1s_d = [max((prf(cand['mask'], gt)[2] for cand in raw_cands), default=0.0)
                      for gt in gt_test.values()]
            if f1s_d:
                print(f'  测试社区（{len(f1s_d)} 个）best-F1 均值 {np.mean(f1s_d):.3f}  '
                      f'>0.5 检出 {sum(f > 0.5 for f in f1s_d)}')

        # 阈值标定：候选分数分布 + 多档扫描（复用候选，无额外前向）
        # tau 应按模型自身分数标度选取，不同数据集/训练配置下分布差别很大。
        if raw_cands and gt_test:
            q = np.asarray([c['quality'] for c in raw_cands])
            stab = np.asarray([c['stability'] for c in raw_cands])

            def pct(a):
                return '/'.join(f'{np.percentile(a, p):.3f}' for p in (10, 50, 90))

            print(f'  候选分布（{len(raw_cands)} 条，过滤前）：'
                  f'quality p10/50/90 {pct(q)} max {q.max():.3f}；'
                  f'stability p10/50/90 {pct(stab)} max {stab.max():.3f}')
            print('  == 阈值扫描（AvgF1=CLARE 口径；最后一行 = 当前 --tau-q/--tau-s）==')
            print(f'  {"tau_q":>6s} {"tau_s":>7s} {"kept":>5s} {"found":>7s} '
                  f'{"AvgF1":>7s} {"bestF1":>7s} {"cover":>6s}')
            for tq, ts_ in (0.0, 0.0), (0.1, 0.02), (0.2, 0.05), (0.2, 0.2), \
                    (0.33, 0.3), (args.tau_q, args.tau_s):
                ks = dedup_cands([c for c in raw_cands
                                  if c['quality'] >= tq and c['stability'] >= ts_],
                                 args.tau_j, key=args.dedup_key)
                f1s_s = [max((prf(k['mask'], gt)[2] for k in ks), default=0.0)
                         for gt in gt_test.values()]
                f1p_s = [max((prf(k['mask'], gt)[2] for gt in gt_test.values()),
                             default=0.0) for k in ks]
                avgf1 = (0.5 * (float(np.mean(f1s_s)) + float(np.mean(f1p_s)))
                         if f1s_s and f1p_s else 0.0)
                n2p = -np.ones(n, dtype=np.int64)
                for pi, k in enumerate(ks):
                    for v in k.get('nodeset', ()):
                        if n2p[v] < 0:
                            n2p[v] = pi
                cov = float((n2p >= 0).mean())
                print(f'  {tq:6.2f} {ts_:7.3f} {len(ks):5d} '
                      f'{sum(f > 0.5 for f in f1s_s):3d}/{len(f1s_s)} '
                      f'{avgf1:7.3f} {np.mean(f1s_s):7.3f} {cov:6.3f}')
        results_acg[mode_name] = dict(
            f1_true=float(np.mean(f1_true)) if f1_true else 0.0,
            f1_pred=float(np.mean(f1_pred)) if f1_pred else 0.0,
            coverage=float(covered.mean()), nmi=float(nmi), ari=float(ari),
            # 主指标：未过滤集合（与 CLARE/ProCom 可比）
            sym_f1=clare_raw['sym_f1'], sym_j=clare_raw['sym_j'],
            onmi=clare_raw['onmi'], detect_pct=clare_raw['detect_pct'],
            found=clare_raw['found'], det_p=clare_raw['det_p'],
            det_r=clare_raw['det_r'], det_f1=clare_raw['det_f1'],
            det_matched=clare_raw['det_matched'],
            expo_hi_f1_raw=rhi_f1, expo_hi_found_raw=rhi_found,
            expo_lo_f1_raw=rlo_f1, expo_lo_found_raw=rlo_found,
            # 后处理消融：过滤+去重后的集合（本方法实际输出）
            sym_f1_kept=clare_kept['sym_f1'], sym_j_kept=clare_kept['sym_j'],
            onmi_kept=clare_kept['onmi'], detect_pct_kept=clare_kept['detect_pct'],
            found_kept=clare_kept['found'],
            expo_hi_f1=hi_f1, expo_hi_found=hi_found, expo_hi_n=hi_n,
            expo_lo_f1=lo_f1, expo_lo_found=lo_found, expo_lo_n=lo_n,
            cov_hi_f1=chi_f1, cov_hi_found=chi_found, cov_hi_n=chi_n,
            cov_zero_f1=cze_f1, cov_zero_found=cze_found, cov_zero_n=cze_n,
            det_kept=len(kept), det_raw=len(raw_cands), det_gt=len(gt_test))
    return results_acg


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', required=True,
                    help='sampleQuery.py 生成的 *_queries.pt')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default=None,
                    help='GPU 设备，如 cuda:0 / cuda:1；不传则沿用默认 cuda:1')
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--frozen-h', default=None,
                    help='冻结编码器：直接加载预计算的 H（potion.pretrain 输出），'
                         '训练只更新提示编码器与解码器。大图（Amazon2M）用这条路径避免全图卷积')
    ap.add_argument('--layers', type=int, default=3, help='编码器层数')
    ap.add_argument('--rounds', type=int, default=2,
                    help='交互训练轮数（第 1 轮主监督，后续轮错误驱动补点后辅助监督）')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                    help='随机种子列表：逐个训练+评估；多 seed 时自动汇总 mean±std')
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--add-max', type=int, default=2,
                    help='交互轮每轮最多新增的错误驱动提示点数')
    ap.add_argument('--interact-weight', type=float, default=1.0,
                    help='交互轮损失的权重（第一轮权重固定为 1）')
    ap.add_argument('--p-drop', type=float, default=0.7,
                    help='首轮随机子集的 token 丢弃率；--prompt-mode dropout 时'
                         '作用于全部训练提示')
    ap.add_argument('--prompt-mode', choices=['mix', 'dropout'], default='mix',
                    help='首轮提示采样：mix=完整/仅正/单种子/随机子集各约 1/4；'
                         'dropout=旧版逐 token 随机丢弃')
    ap.add_argument('--drop-neg-train', action='store_true',
                    help='消融：训练时剥掉**所有**负 token（负点与负属性），覆盖两条支路 ——'
                         '多轮交互的主提示（含交互轮 FP→负点 的补点）与**属性辅助任务**'
                         '（后者在循环之外、有自己的 decode，首版漏过它）。保留正属性：'
                         '"要不要属性监督"与"要不要负提示"是两个问题，一起动就不是单变量。'
                         '判据：若推理时 +neg 仍高于 base，说明模型本来就学到了，而非靠负提示；'
                         '日志"带负点 x/y"在这个开关下应当显示 0（现在数的是剥后口径）。')
    ap.add_argument('--hide-ratio', type=float, default=0.2,
                    help='社区隐藏 episode：每 batch 随机隐藏该比例的训练社区'
                         '（其查询跳过监督），打破"记忆社区成员"的塌缩')
    ap.add_argument('--feat-drop', type=float, default=0.15,
                    help='节点特征 dropout：训练时随机置零特征非零项，'
                         '破坏节点 ID 记忆，逼模型用属性/结构模式')
    ap.add_argument('--attr-aux', type=float, default=0.3,
                    help='属性-only 辅助损失权重：直接训练"属性提示 → 圈掩码"映射'
                         '（0 = 关闭）')
    ap.add_argument('--synth-k', type=int, default=160,
                    help='合成伪圈增广：生成的合成圈数量（0 = 关闭；'
                         '圈数×2 条合成查询仅加入训练池，val/test 永远真实圈）')
    ap.add_argument('--synth-per-real', type=float, default=0.5,
                    help='每轮抽取的合成查询数/真实训练查询数；0.5=真实:合成 2:1，'
                         '1=1:1，2=1:2；默认使用全部合成查询')
    ap.add_argument('--synth-attr-max', type=int, default=64,
                    help='属性聚类圈的最大节点数；真实圈普遍 16~30 节点时建议降到'
                         '64 左右，避免合成与真实目标尺度错位')
    ap.add_argument('--synth-source', choices=['perturb', 'struct', 'both'],
                    default='struct',
                    help='除属性聚类圈外，另一半合成圈从哪来：perturb=训练圈扰动/重组'
                         '（旧行为，用于消融对照）；struct=只用图结构、不依赖任何社区'
                         '标签（PPR 锥/贪心扩张/BFS 球），对症"未见社区定位不准"；'
                         'both=两者对半（默认）')
    ap.add_argument('--synth-struct-min', type=int, default=16,
                    help='结构性合成圈的最小节点数（真实社区池口径是 ≥16）')
    ap.add_argument('--synth-struct-parts', default='ppr,expand,ball',
                    help='结构性圈用哪些生成器（逗号分隔，轮转提供候选）：'
                         'ppr=个性化 PageRank 锥，expand=贪心凝聚扩张，ball=BFS 球。'
                         '生成器只负责候选多样性，分布对齐由选样完成')
    ap.add_argument('--synth-struct-cand', type=int, default=6,
                    help='每个槽位生成几个候选、取 conductance 最接近目标分布的那个'
                         '（1 = 关闭匹配；实测 6 可把 cond 的 IQR 从 0.37 压到 0.17）')
    ap.add_argument('--synth-reject-j', type=float, default=0.5,
                    help='结构性合成圈的拒绝阈值：与任一 val/test 社区 Jaccard 超过'
                         '该值即丢弃，堵住"结构圈恰好等于某个评测社区"的软泄露')
    ap.add_argument('--interact-rounds', type=int, default=5)
    ap.add_argument('--mask-head', choices=['dot', 'mlp'], default='dot',
                    help='掩码头：dot=点积 ⟨h,e⟩/√d（默认，当前行为）；'
                         'mlp=点积 + 残差 MLP([h⊙e, |h−e|])，末层零初始化 → 训练起点'
                         '与 dot 逐位相同，用于检验"提高节点级判别分辨率"能否改善检索'
                         '（诊断显示瓶颈是"排不出谁属于社区"，不是"不知道在哪"）')
    ap.add_argument('--shape', choices=['none', 'trim', 'lcc'], default='none',
                    help='输出掩码的形状规则（检索与检测统一走这一个开关）：'
                         'lcc=取最大连通分量（默认；真实社区 95% 是一个连通块，'
                         '实测节点+属性侧 AvgF1 0.240→0.277）；'
                         'trim=只去孤立边缘；none=原样（可逐位复现旧行为）')
    ap.add_argument('--loss', choices=['bce_dice', 'tversky'], default='bce_dice',
                    help='重叠项：bce_dice=旧口径 BCE+Dice（默认，逐位复现历史日志）；'
                         'tversky=BCE + Tversky(α>β 惩罚误检)。动机：pos_weight 顶到 50 '
                         '使漏检代价是误检的 50 倍，最优决策点 p≈0.02 而非 0.5，'
                         '实测操作点 P 0.180 / R 0.494 且 oracle-size 恒低于 0.5 阈值')
    ap.add_argument('--pw-max', type=float, default=50.0,
                    help='BCE pos_weight 的上限。默认随 --loss 而定：bce_dice→50（旧口径）、'
                         'tversky→1（中性，去掉召回偏置）。显式传值可做单变量消融，'
                         '例如 --loss bce_dice --pw-max 1 只去掉偏置、不换重叠项')
    ap.add_argument('--tv-alpha', type=float, default=0.7,
                    help='Tversky 的 **FP（误检）权重**；>β 则更恨误检 → 集合更紧（提高精度）')
    ap.add_argument('--tv-beta', type=float, default=0.3,
                    help='Tversky 的 **FN（漏检）权重**。α=β=0.5 时 Tversky 恒等于 Dice'
                         '（可作等价性对照）')
    ap.add_argument('--prior', choices=['none', 'ppr'], default='none',
                    help='查询局部逐节点先验（方案 2+3）：none=旧行为（默认，逐位复现）；'
                         'ppr=logits += γ·正点PPR锥 − λ·负点PPR锥（γ/λ 可学、初值 0）。'
                         '诊断依据：正点锥质心系统性偏向混淆社区 B（同号比例 32.5%），'
                         '而"正锥−λ·负锥"把 A 的 Dice 从 0.427 抬到 0.461')
    ap.add_argument('--prior-neg', choices=['off', 'cone'], default='cone',
                    help='先验的负半边：cone=用负点的 PPR 锥作减项（默认，= 方案 2）；'
                         'off=只用正锥（= 方案 3 单独，作消融）')
    ap.add_argument('--prior-alpha', type=float, default=0.15,
                    help='PPR 的跳转概率 α（与 synth 侧同默认）')
    ap.add_argument('--prior-iters', type=int, default=20,
                    help='PPR 幂迭代次数（收敛判据 1e-10，通常远早于该值就停）')
    ap.add_argument('--prior-lambda', type=float, default=None,
                    help='把 λ（负锥信任度）**钉死**在该正值、不参与学习。'
                         '动机：45/46 实测 λ 自由学习时有 2/3 个 seed 学到**负值**'
                         '（= 把负锥加进去、方向反了），且 λ 的符号与任何指标都不相关 —— '
                         '说明自由 λ 没有找到 Q5 测出的那个减项。Q5 显示 λ≈0.5~1 时'
                         '锥对 A 的 Dice +0.034，这里就是用来验证"钉住符号后能否迁移到解码器"。'
                         '不给则 λ 自由学习（旧行为）')
    ap.add_argument('--cand', choices=['full', 'ball', 'auto'], default='auto',
                    help='候选集口径：full=全图解码（默认）；ball=种子 ball 局部解码，'
                         '只解候选集并把 logits 散射回全图（换搜索速度，改变召回上限）')
    ap.add_argument('--cand-thresh', type=float, default=0.85,
                    help='--cand auto 的判定阈值：train 覆盖率 >= 该值就用 ball，否则全图')
    ap.add_argument('--cand-hop', type=int, default=2, help='ball 半径（默认 2 跳）')
    ap.add_argument('--cand-cap', type=int, default=15000, help='候选集上限，按 BFS 顺序截断')
    ap.add_argument('--skip-detect', action='store_true',
                    help='跳过检测（ACG）评估，只跑搜索；用于快速拿搜索指标')
    ap.add_argument('--M', type=int, default=1000,
                    help='ACG 种子数；0 或 ≥节点数 = 每个节点都做种子（消除覆盖瓶颈，'
                         '解码仅多几秒；论文建议用此档，避免"小社区没被采到"的假阴性）')
    ap.add_argument('--acg-seed', choices=['uniform', 'partition'], default='uniform',
                    help='种子来源：uniform=均匀随机（默认；已实测优于 partition）；'
                         'partition=按结构划分（标签传播）max-min 分配，仅作消融对照')
    ap.add_argument('--acg-attr', type=int, default=3,
                    help='ACG 种子附带的属性提示数（种子自身 top 特征；0 = 仅节点提示）')
    ap.add_argument('--tau-q', type=float, default=0.5)
    ap.add_argument('--tau-s', type=float, default=0.6)
    ap.add_argument('--tau-j', type=float, default=0.7)
    ap.add_argument('--dedup-key', choices=['quality', 'stability'], default='quality',
                    help='去重的贪心排序键：quality=旧行为；stability=按 SAM 式稳定性'
                         '排序（规则对照中与宽阈值 q≥0.33 组合最优）')
    ap.add_argument('--save', action='store_true',
                    help='（可选）保存 best.pt/config.json 供复用；默认不产生中间文件')
    ap.add_argument('--k-gran', type=int, default=1,
                    help='粒度 token 数 K（方案 4）。1=旧行为（单掩码，逐位复现）；'
                         'K>1 时每个粒度 token 认领一个**候选社区**（= 种子点所属的多个社区，'
                         '见 build_alt_targets），质量头当选择器、部署时取它的 top-1。'
                         '动机：种子点是刻意选的重叠点 → 提示本身有歧义，单掩码的最优解必然是'
                         '折中集；SAM 对歧义的原生答案就是多掩码 + IoU 头')
    ap.add_argument('--restrict-hop', type=int, default=0,
                    help='**输出规则**：把预测掩码限制在提示正点的 k 跳邻域内（0=关闭，默认，'
                         '逐位等价旧行为）。动机来自基础能力诊断：plain 组有 80%% 的掩码落在'
                         '种子点 2 跳之外，而 plain 的答案就在种子身边 —— 那 80%% 正是'
                         '"检索了远处的记忆社区"的结构指纹。无标签（只用图结构与提示）。'
                         '顺序是先按球限制、再走 --shape')
    ap.add_argument('--pos-mode', choices=['off', 'feat', 'head'], default='feat',
                    help='**相对位置通道的注入方式**（阶段12；默认 off，逐位等价旧行为）。'
                         'feat=把 r(v|提示正点) 加进解码器的节点表示（对应 SAM 的 '
                         'src = src + image_pe，最像 SAM 的那种接法）；head=把逐 (节点, '
                         '粒度) 的 [点积, r] 喂给一个零初始化的小 MLP、再残差加回 logits。'
                         '两者末层都零初始化 → 第一步前向与 off **逐位相同**，训练差异只能'
                         '来自学到的部分。动机：掩码是 ⟨h_v,e⟩/√d，对 (节点, 提示) 的**关系**'
                         '毫无依赖 → 模型只能检索记忆里的社区（98%% 查询的最佳匹配不是目标），'
                         '种子邻域内概率≈0（--restrict-hop 一限制就整片变空）')
    ap.add_argument('--hop-mode', choices=['khop', 'full'], default='khop',
                    help='hop 距离表口径：khop=只展开 <=buckets-1 跳（默认，快）；'
                         'full=精确最短路后再截断（慢，buckets 很大时才不同）')
    ap.add_argument('--pos-chan', choices=['hop', 'ppr', 'both'], default='hop',
                    help='r 的定义：hop=到提示正点的跳距（5 桶 one-hot，预计算全源距离表）；'
                         'ppr=归一化 log-PPR（连续、带流的信息；逐提示算 + 有上限的缓存）；'
                         'both=两者拼接（6 维）。参考点只取**正点**（负点/属性不参与）；'
                         '**无标签**（只用图结构 + 提示）→ 训练/测试同构。位置只在解码器里'
                         '用，编码器保持提示无关（CAM 对齐 SAM 的核心）')
    ap.add_argument('--attr-mode', choices=['off', 'sim'], default='sim',
                    help='**属性–节点相似度通道**（阶段45；默认 off，逐位等价旧行为）。'
                         'sim=把"节点属性向量 × 提示属性向量"的余弦相似度**直接加到 logits**'
                         '（A 行归一化；q=正属性均值 − 负属性均值，再归一化），强度由 '
                         '--attr-w 控制。动机（cora_nosynth）：`attr`（纯属性提示）与 `full` '
                         '给出完全相同的 F1/掩码 ⇒ 属性 token 只经 transformer 间接影响 query、'
                         '从未进入算掩码的那一步；而两边的社区/类别是词袋可分的。'
                         '无标签、训练/测试同构；无属性 token 的提示（n/n-n）该场恒为 0')
    ap.add_argument('--attr-w', type=float, default=1.0,
                    help='属性相似度场的**初始权重**（仅 --attr-mode sim 时建这个参数）。'
                         '场已在每个查询内做过**中心化+标准化**（零均值、单位方差），'
                         '所以 w=1 ≈ 每个标准差给 1 个 logit；w=0 时起点与 off 逐位等价。'
                         '训练结束后会打印收敛值（判读：≈0 → 没用上；持续变大 → 在依赖它）')
    ap.add_argument('--attr-drop', type=float, default=0.0,
                    help='**属性场 dropout**（阶段46；默认 0.0，逐位等价旧行为）。训练时按提示以'
                         '概率 p 把整行属性场置零，迫使模型在"无场"条件下也能出掩码；推理时场恒开。'
                         '动机：Cora 上开属性通道后 node-only 从 0.245 掉到 0.070 —— 属性通路把'
                         '节点通路饿死了。加 0.3–0.5 的 dropout 可以让两条通路同时保留'
                         '（代价通常是 full 掉 0.01–0.02，换 node-only 回来 0.1 以上）')
    ap.add_argument('--seed-sim', choices=['off', 'mean', 'max', 'both'], default='off',
                    help='**种子内容相似度通道**（阶段37；默认 off，逐位等价旧行为）。'
                         '在相对位置通道之后**再拼一块**：cos(H_v, mean_{正点} H)（mean）、'
                         'max_{正点} cos(H_v, H_s)（max）或两块都要（both），H = 编码器输出'
                         '（内部 detach）。动机（离线体检 temp/prompt_src_diag.py，220 条 test）：'
                         'hop + 0.25·相似度把 P@|A| 从 0.353 提到 **0.400**（配对 127胜/58负），'
                         '且受益在 unseen 种子（+0.046）与"成员重叠<0.3"（+0.040）两组同样出现 '
                         '⇒ 不是记忆捷径；随机种子/随机属性对照只有 0.018~0.020。'
                         '⚠ 是**叠加**不是替换：单独用相似度场更差（P@|A| 只与 hop 平手）。'
                         '⚠ 与 --pos-mode off 不兼容（同一个注入点，main 里 ap.error 拦）。'
                         '⚠ 属性侧的对应物（属性 token 换成其携带者的 H 池化）实测死透，未实现')
    ap.add_argument('--prompt-src', choices=['raw', 'H'], default='H',
                    help='**提示 token 的来源**（阶段38；默认 raw，逐位等价旧行为）。'
                         'raw=原始输入特征（node_proj(feat[id])，对齐 SAM 的模块分工：提示'
                         '编码器与图编码器解耦）；H=图编码后的表示（node_proj_h(H[id])，'
                         '提示侧并进图编码器 ⇒ 梯度从提示 token 回流进编码器）。'
                         '两种模式下**属性 token 都是 attr_emb 查表，不变**。'
                         '离线体检（temp/prompt_src_diag.py，冻结的**代理 H**）里"把节点 token '
                         '换成 H[v]"单独用与 hop 场≈平手（P@|A| 0.387 vs 0.353、配对 106胜/91负）'
                         '——本开关就是让它在**真 H** 下端到端训练一次，看打通梯度路径后是否不同。'
                         '⚠ 与 --anon-seed-p 互斥（后者要求提示 token 保留原始信息，见 ap.error）')
    ap.add_argument('--pos-neg', choices=['off', 'hop', 'ppr', 'both'], default='off',
                    help='**负提示的相对位置通道**（阶段14；默认 off，逐位等价旧行为）。'
                         'r_neg(v) = 到负提示点的跳距/PPR，与 --pos-chan 同一套定义与归一化，'
                         '作为**另一块拼在**正通道后面（注入方式由 --pos-mode 决定）。'
                         '动机：假阳性质量分解显示 51%% 的 FP 质量是种子点的**其它社区 B\\A**'
                         '（负提示点就在那里），而负提示目前几乎没用（增益 all +0.001）——'
                         '信息在提示里、模型拿不到。方案 3 把同一个量做成**加性标量场**'
                         'γ·w_pos − λ·w_neg 而失败；本通道把它做成解码器的**输入特征**，'
                         '即"失败在注入方式、不在那个量"的直接检验。无标签、训练+评测同构')
    ap.add_argument('--neg-keep', choices=['mix', 'all'], default='mix',
                    help='训练提示里**负点的保留率**（阶段16；默认 mix = 现状，逐位等价）。'
                         'mix：负点只在 full 形态（≈1/4）与 subset 侥幸存活时出现，实测只有约'
                         '1/3 的训练提示带负点；而**部署时每条提示都带**（make_prompt 恒给负点）。'
                         'all：任何形态都补回该查询的全部负点，覆盖≈100%、与部署口径一致。'
                         '不变式：只加不减、不动正点侧、不额外消耗随机数（因此与 mix 的差异'
                         '**只能是负点覆盖**）。动机：① B\\A 占 FP 质量 ~51%% 不降，而负提示增益'
                         'all 仅 +0.001、负参照几何通道（阶段14）也无效 → 先判定这是'
                         '"覆盖率问题"还是"模型学不会"。日志里看"带负点 x/y"确认开关生效。')
    ap.add_argument('--attr-only-p', type=float, default=0.0,
                    help='**纯属性提示**的训练概率（阶段23；默认 0.0 = 不改变任何行为，'
                         '连随机数都不多抽一个）。以该概率把当前这条训练提示换成"纯属性"'
                         '形态：去掉**全部节点 token**、只留属性 token（正 + 负）。'
                         '动机（output68 的 `attr` 行动读数）：纯属性提示 F1 仅 0.049、'
                         '掩码均值 3.5 个节点（GT 53.6），而 base 0.314；ACG 侧 n+a 也只比 '
                         'node-only 高 +0.006 ⇒ 检索分数**全部**来自种子点。属性取自 '
                         'community_attributes（社区 top-3、共享属性优先）＝**社区级语义'
                         '画像**，是唯一能迁移到"训练集里没有的社区"的输入。'
                         '⚠ 注意与既有 `--attr-aux`（默认 0.1）的关系：那条辅助任务**本来就**'
                         '对每条查询每轮算一次纯属性损失（同一构造）—— 本开关是把同一条提示'
                         '放进**主任务**（该样本权重 1.0 而非 0.1）并**进入交互轮**，力度更大；'
                         '想先轻试就直接抬 --attr-aux。两者读数都看检索表的 `attr` 行。'
                         '判据（家族 sd：检索 0.026 / ACG n+a 0.039）⇒ 每臂 ≥3 seed、'
                         '余量 ≥+0.05（检索）/ +0.07（n+a）。')
    ap.add_argument('--scale', choices=['off', 'fit'], default='off',
                    help='**逐查询尺度**（阶段20；默认 off = 不改变任何行为）。'
                         'fit：在**训练查询**上算 oracle-T（9 档里 F1 最好的那档），岭回归拟合 '
                         'T̂ = w·φ(x)+b（φ 全是无标签特征：概率场尺寸曲线 + 提示/2 跳邻域规模），'
                         '然后把 T=0 / T̂ / oracle-T 三条口径**同 run 配对**打出来（val/test + '
                         'ACG）。判据：T̂ ≈ T=0 → 学不动、放弃；T̂ 接近 oracle-T → 值得保留。'
                         '⚠ 部署时它是一个离线固定函数（换数据集要重新拟合）')
    ap.add_argument('--scale-lambda', type=float, default=1.0,
                    help='岭回归的 λ（特征已标准化）；默认 1.0')
    ap.add_argument('--alt-neg-w', type=float, default=0.0,
                    help='**B\\A 选择性抑制**（阶段24；默认 0.0 = 不改变任何行为，日志也逐字节'
                         '不变）。对"种子点的**其它社区** B 减去自身社区 A"的那些节点，额外加一项 '
                         'w · BCE(logits→0)；等价于把这些节点的负类权重从 1 提到 1+w，别的节点'
                         '一字不动。动机：B\\A 稳定占 **FP 质量 48~49.5%%**（三次输入侧机制——'
                         '加性先验/负参照几何通道/负点覆盖率——都没动过它），而"完美清掉 B\\A 且 '
                         'TP 不变"的算术上界是检索 0.316→0.48（+0.16）；塌缩诊断又显示 84~85%% 的'
                         '最佳匹配落在训练社区 ⇒ 模型现在的答案就是"种子点所属的训练社区"。'
                         '与已失败的两类机制的区别：输入侧喂信息（模型拿不到）、全局收紧（不分'
                         '对象地压负类）—— 本项是**输出侧的选择性监督**。B\\A 与 A 恒不相交'
                         '（掩码显式减掉 A），故不会惩罚真阳性。'
                         '**量纲与先扫哪几档**：本项是该查询 B\\A 节点上的 BCE **均值**（不是和），'
                         '与主损失的逐查询量级相当（≈0.5），所以 w=1 大致等于"给这些节点再叠加'
                         '一份同规模的负类损失"。建议先扫 **0.2 / 1**，再大就接近"用这一项压过'
                         '主损失"了。'
                         '读数：日志里 `B\\A加压 x/y`（覆盖面）、FP 质量分解里 B\\A 的占比、'
                         '以及"P↑ 而 R 基本不动"。反面判据：R 明显掉 + 掩码整体变小 ⇒ 那只是'
                         '全局收紧重演，判负。')
    ap.add_argument('--weight-decay', type=float, default=1e-4,
                    help='优化器的 weight decay（阶段24；默认 1e-4 = 与之前**硬编码**的值相同，'
                         '逐位等价）。之前它是写死在 AdamW 里的，本开关只是把它变成可扫的旋钮。'
                         '用途：反记忆（轨迹显示后期是排序退化、真覆盖中位早死）。')
    ap.add_argument('--anon-seed-p', type=float, default=0.0,
                    help='**种子点特征匿名**（阶段27 路 A；默认 0.0 = 不改变任何行为，连随机数都'
                         '不多抽一个）。以该概率把一条训练查询的**正种子点**行在**编码器输入**里置零，'
                         '而**提示通道仍用原始特征** ⇒ "用户点了哪个点"完整保留、"这个点的身份"'
                         '不可查。动机（阶段26 的结构性发现）：假阳性的两个成分（训练社区 / 种子点'
                         '其它社区 B\\A）**互相顶替、合计守恒在 FP 质量的 75~84%%**（15 次运行），'
                         '所以"只削一个成分"的机制（输入侧喂信息、输出侧选择性惩罚、增广打记忆）'
                         '注定只换来份额重排；而**捷径**是"编码器读得出节点身份 ⇒ 背出它所属的'
                         '训练社区"（84~85%% 的最佳匹配落在训练社区）。本开关直接堵这条捷径，'
                         '是 `--feat-drop`（全局随机丢）的**定向版**、与 `--hide-ratio`（整社区'
                         '不进 batch）不同：数据没少，只是身份不可读。'
                         '⚠ 只匿名正种子点（负点是提示信息，不是捷径）；**评测期不匿名**（与 '
                         'feat_drop 同构的训练期正则）。'
                         '**预登记判据**：`种子点其它社区(B\\A)` + `训练社区` 两项**合计**占 FP 质量'
                         '必须下降（基线 78.7%），只看一个成分会重蹈阶段24/26；'
                         '并配 检索 ≥+0.05 / ACG n+a ≥+0.07（阶段22 尺子）。日志里 `匿名 x/y` 确认生效。')
    ap.add_argument('--prompt-cats', choices=['taxonomy', 'legacy'],
                    default='taxonomy',
                    help='训练提示的形态集（阶段31）。taxonomy=**8 个提示类别**，规则是'
                         '**正节点必备**：n / n+a / n-n / n-a / n+a-n / n+a-a / n-n-a / full'
                         '（n=正节点、+a=正属性、-n=负节点、-a=负属性），"带负"的 6 个类别'
                         '合计占 --neg-cover。目的是把负信息拆成**两条可单变量测量**的通道'
                         '（旧 make_prompt 总是负点+负属性一起给 ⇒ '
                         '"负节点没用"与"负属性没用"从未分开验证过）。'
                         'legacy=旧的四形态采样（full/positive/seed/subset），逐位复现历史日志。'
                         '⚠ taxonomy 下 --p-drop/--prompt-mode dropout/--neg-keep 都不生效'
                         '（抽稀会让类别不再是类别），与 --drop-neg-train 互斥。')
    ap.add_argument('--neg-cover', type=float, default=2.0 / 3.0,
                    help='taxonomy 模式下"带负类别"的采样权重合计（默认 2/3：无负两类各 1/6、'
                         '带负六类各 1/9）。默认留 1/3 的无负形态，是为了不重演 output59 '
                         '（负点覆盖率 1.00×199 轮时 base 通路被打坏：掩码均值 150→423、'
                         'oracle-size 余量被吃、检测 −0.025）。传 1.0 即复现那种全覆盖。')
    ap.add_argument('--traj-every', type=int, default=5,
                    help='**epoch 轨迹诊断**（阶段21/22）：前 10 轮每轮 + 之后每 N 轮打一行'
                         '`T=0 / mean切 / oracle-size` 三条口径的 F1 + 掩码均值 + 真覆盖中位。'
                         '**只打印、不参与任何选择**。用途：output66 用它定住了"峰值在 epoch 4、'
                         'test 也同意、之后过覆盖回潮"；`mean切`（逐查询把阈值放在字段均值上）是'
                         '**无标签的漂移不变探针** —— 它在后期不跟着掉就说明该实现逐查询标准化。'
                         '成本约 1 分钟/seed。0=关闭')
    ap.add_argument('--out', default='runs/cora')
    args = ap.parse_args()
    if args.synth_per_real is not None and (
            not np.isfinite(args.synth_per_real) or args.synth_per_real < 0):
        ap.error('--synth-per-real 必须是有限的非负数')
    if not np.isfinite(args.p_drop) or not 0 <= args.p_drop <= 1:
        ap.error('--p-drop 必须在 [0, 1] 之间')
    if not np.isfinite(args.neg_cover) or not 0 <= args.neg_cover <= 1:
        ap.error('--neg-cover 必须在 [0, 1] 之间')
    if args.prompt_cats == 'taxonomy' and args.drop_neg_train:
        ap.error('--prompt-cats taxonomy 与 --drop-neg-train 互斥：后者会把 6 个带负类别的'
                 '负 token 全剥掉，类别退化成 2 种（= output83 那次的无效设置）。'
                 '要做"干净无负模型"请用 --prompt-cats legacy --drop-neg-train。')
    if not (np.isfinite(args.tv_alpha) and np.isfinite(args.tv_beta)
            and args.tv_alpha >= 0 and args.tv_beta >= 0
            and args.tv_alpha + args.tv_beta > 0):
        ap.error('--tv-alpha/--tv-beta 必须是非负且不同时为 0')
    if args.pw_max is not None and (not np.isfinite(args.pw_max)
                                    or args.pw_max < 1.0):
        ap.error('--pw-max 必须是 ≥1 的有限数（pos_weight 不能小于 1）')
    if not (0.0 < args.prior_alpha < 1.0):
        ap.error('--prior-alpha 必须在 (0,1) 之间')
    if args.pos_neg != 'off' and args.pos_mode == 'off':
        ap.error('--pos-neg 需要 --pos-mode feat/head（off 时没有任何地方会用到负通道）')
    if args.seed_sim != 'off' and args.pos_mode == 'off':
        ap.error('--seed-sim 需要 --pos-mode feat/head（相似度块与几何块共用同一个注入点）')
    if args.prompt_src == 'H' and args.anon_seed_p > 0:
        ap.error('--prompt-src H 与 --anon-seed-p 互斥：匿名消融的前提是"提示 token 用原始'
                 '特征、只堵编码器查节点身份"；H 模式下提示 token 就是编码器的输出，'
                 '匿名会把提示一起糊掉，两件事的解释会混在一起')
    if not (np.isfinite(args.scale_lambda) and args.scale_lambda >= 0):
        ap.error('--scale-lambda 必须是非负有限数')
    if not (np.isfinite(args.attr_only_p) and 0.0 <= args.attr_only_p <= 1.0):
        ap.error('--attr-only-p 必须在 [0, 1] 之间（0 = 关闭，逐位等价旧行为）')
    if not (np.isfinite(args.alt_neg_w) and args.alt_neg_w >= 0.0):
        ap.error('--alt-neg-w 必须是非负有限数（0 = 关闭，逐位等价旧行为）')
    if not (np.isfinite(args.anon_seed_p) and 0.0 <= args.anon_seed_p <= 1.0):
        ap.error('--anon-seed-p 必须在 [0, 1] 之间（0 = 关闭，逐位等价旧行为）')
    if not (np.isfinite(args.weight_decay) and args.weight_decay >= 0.0):
        ap.error('--weight-decay 必须是非负有限数（默认 1e-4 与旧硬编码值相同）')
    if args.traj_every < 0:
        ap.error('--traj-every 必须 ≥0（0 = 关闭轨迹诊断）')
    if args.prior_iters < 1:
        ap.error('--prior-iters 必须 ≥1')
    if args.prior_lambda is not None:
        if args.prior == 'none' or args.prior_neg != 'cone':
            ap.error('--prior-lambda 需要 --prior ppr 且 --prior-neg cone')
        if not np.isfinite(args.prior_lambda) or args.prior_lambda <= 0:
            ap.error('--prior-lambda 必须是正有限数（负值 = 把负锥加进去，方向反了）')
    # 把 None 解析成实际生效值，让配置头/json 里记录的是真实口径而不是 None
    args.pw_max = resolve_pw_max(args.loss, args.pw_max)
    device = (args.device if getattr(args, 'device', None) else
          ('cuda:1' if torch.cuda.is_available() else 'cpu'))

    # 运行配置头：让每份日志自带完整参数
    print('== 运行配置（CLI 参数）==')
    for k, v in vars(args).items():
        print(f'  {k:16s} = {v}')
    # 提示形态 / 评估表口径（阶段31）：类别表改了什么、哪些开关在类别模式下不生效，
    # 必须让日志自带，否则单看一份汇总无法判读（"为什么 p_drop 是 0.7 却没抽稀"之类）。
    if args.prompt_cats == 'taxonomy':
        print(f'== 提示类别 ==  taxonomy：8 类（正节点必备 n / n+a / n-n / n-a / n+a-n / '
              f'n+a-a / n-n-a / full），带负类别共占 neg_cover={args.neg_cover:.2f}'
              f' ⇒ 无负形态 {100 * (1 - args.neg_cover):.0f}%'
              '；⚠ p_drop / --prompt-mode / --neg-keep 在类别模式下不生效'
              '（抽稀会让类别不再是类别）')
    else:
        print('== 提示类别 ==  legacy：旧的四形态采样（full/positive/seed/subset）'
              '；⚠ 检索表仍是 8 类别 + rand + attr（评估口径与训练形态无关）')
    print('== 检索表口径 ==  8 类别 + rand（同数量随机负 token 的对照）+ attr（纯属性诊断行）'
          '；`full` / `n+a` 逐位等于旧的 +neg / base')
    print(f'== 损失 ==  {args.loss}（BCE pos_weight 上限 {args.pw_max:.1f}'
          + (f'，Tversky α(FP)={args.tv_alpha} β(FN)={args.tv_beta}'
             f' → {"精度侧" if args.tv_alpha > args.tv_beta else "召回侧"}）'
             if args.loss == 'tversky' else '）'))
    if args.prior != 'none':
        print(f'== 查询局部先验 ==  {args.prior}（负半边 '
              f'{"开" if args.prior_neg == "cone" else "关"}；'
              f'α={args.prior_alpha} iters={args.prior_iters}）；'
              'γ/λ 初值 0、可学习，收敛值在每个 seed 训练结束后打印')

    data = load_raw(args.queries, device)
    if getattr(args, 'cand', 'full') == 'auto':
        args.cand = _auto_cand(args, data)
    print(f'== 图与查询 ==  节点 {data["n"]}  属性维度 {data["num_attrs"]}  '
          f'边 {data["edge_index"].size(1) // 2}  '
          f'社区 {len(data["g"]["communities"])}  查询 {len(data["qs"])}')

    all_res = []
    all_sc = []                       # [(seed, scale_report(...))]；--scale off 时为空
    pos_dim = pos_dim_total(args) if args.pos_mode != 'off' else 0
    if args.pos_mode != 'off':
        _neg_txt = (f'，负参照 {args.pos_neg}' if args.pos_neg != 'off' else '')
        print(f'== 相对位置 ==  mode={args.pos_mode}  正参照 {args.pos_chan}{_neg_txt}'
              f'  维度 {pos_dim}（无标签；训练+评测同时生效）')
        if args.pos_chan in ('hop', 'both'):
            print(f'  ⚠ hop 通道首次用到时预计算全源距离表：{data["n"]}² 字节 ≈ '
                  f'{data["n"] * data["n"] / 1e6:.1f} MB、一次（实测 n≈4 千约 10 秒），'
                  f'之后每条查询只是一次取最小')
    for seed in args.seeds:
        args.seed = seed
        set_seed(seed)
        qs_all = data['qs']
        if qs_all and all('split' in q for q in qs_all):
            # 社区级划分（build_queries.py 生成）：按标记分组，不重划
            data['splits'] = {s: [q for q in qs_all if q['split'] == s]
                              for s in ('train', 'val', 'test')}
        else:
            data['splits'] = split_queries(data['qs'], seed)
        data['synth_qs'] = []
        if args.synth_k > 0:
            from .synth import make_synth_queries
            data['synth_qs'] = make_synth_queries(data, args, seed)
            # ⚠ 合成圈的查询字典由 make_synth_queries 自己造，**不带 `_alts`**。
            # 不补的话它们进训练池后 K 个槽位全是 padding（一点分割梯度都没有），
            # 而且是静默的 —— 首跑就踩了（每轮约 80 条合成查询白跑）。这里补上，
            # 用与真实查询同一道防泄漏白名单（data['alt_allowed']）。
            if args.k_gran > 1:
                for _q in data['synth_qs']:
                    _q['_alts'] = build_alt_targets(_q, data['n2c'],
                                                    data['comm_mask'],
                                                    allowed=data['alt_allowed'])
            print(f'合成圈增广：{args.synth_k} 个合成圈 → '
                  f'{len(data["synth_qs"])} 条合成查询（仅加入训练池）')
        _frozen_H = None
        if getattr(args, 'frozen_h', None):
            _fh = torch.load(args.frozen_h, map_location='cpu')
            _frozen_H = _fh['H'].to(device)
            assert _frozen_H.size(0) == data['n'], (
                f"--frozen-h 的节点数 {_frozen_H.size(0)} 与图 {data['n']} 不一致")
            print(f'== 冻结编码器 ==  H 来自 {args.frozen_h}：{tuple(_frozen_H.shape)}'
                  f'（训练只更新提示编码器与解码器）')
        model = CAM(in_dim=data['feat'].size(1),
                    num_attrs=data['num_attrs'], d=args.d,
                    layers=args.layers,
                    k_gran=args.k_gran,
                    mask_head=args.mask_head, prompt_src=args.prompt_src,
                    pos_mode=args.pos_mode, pos_dim=pos_dim,
                    attr_mode=args.attr_mode, attr_init=args.attr_w,
                    frozen_H=_frozen_H).to(device)

        print(f'\n===== seed {seed} =====')
        print(f'== 训练 ==（{device}）')
        train(model, data, args, device)
        if args.prior != 'none':
            # 预登记读数 1：γ/λ 的收敛值。0 → 先验没被用上；明显非 0 → 模型在用。
            gw, lw = (float(model.decoder.prior_w[0]),
                      float(model.decoder.prior_w[1]))
            # 缓存条数用于估算**主机内存**：每条 (2,n) float32 ≈ 32KB（n=3959）。
            # ⚠ 首跑实测会长到 9.8 万条 ≈ 3.1 GB（训练提示每轮重抽子集 + 交互补点带入
            # 下一轮，键空间爆炸），所以缓存刻意放在 CPU 而非显存。
            print(f'== 先验权重 ==  γ(正锥)={gw:+.4f}  λ(负锥)={lw:+.4f}'
                  f'  → {"在用" if max(abs(gw), abs(lw)) > 0.1 else "几乎没用上"}'
                  f'  （缓存 {len(data["prior_cache"])} 条'
                  f' ≈ {len(data["prior_cache"]) * 2 * data["n"] * 4 / 1e6:.0f} MB 主机内存）')
        if args.attr_mode != 'off':
            # 预登记读数 2：attr_w 的收敛值。≈0 → 属性场没被用上（同 prior_w 的判读方式）。
            aw = float(model.decoder.attr_w)
            print(f'== 属性通道 ==  attr_w = {aw:+.4f}'
                  f'  → {"在用" if abs(aw) > 0.1 else "几乎没用上"}'
                  f'（初值 {args.attr_w:+.2f}；场=属性余弦的逐查询标准化值）')

        _t_ev0 = time.time()
        res = evaluate(model, data, args, device)
        _n_q = len(data['splits']['test'])
        _n_forms = len(PROMPT_CATS) + 2          # 8 类 + rand + attr
        _dt = time.time() - _t_ev0
        print(f'== 搜索推理 == {_dt:.1f}s  平均每查询 {_dt / max(1, _n_q * _n_forms):.2f}s'
              f'（test 查询 {_n_q} × 提示形态 {_n_forms} = {_n_q * _n_forms} 次推理）', flush=True)
        if getattr(args, 'skip_detect', False):
            acg = None
            print('== 检测已跳过 ==  --skip-detect（只跑搜索）')
        else:
            _t_ac0 = time.time()
            acg = acg_eval(model, data, args, device)
            _dt_a = time.time() - _t_ac0
            _n_seed = min(args.M, data['n']) if args.M > 0 else data['n']
            print(f'== 检测推理 == {_dt_a:.1f}s  平均每种子 {_dt_a / max(1, _n_seed):.3f}s'
                  f'（种子 {_n_seed}，两种形态合计）', flush=True)
        # 逐查询尺度（阶段20）：拟合在**训练查询**上做，报告是同 run 配对的三条口径。
        # ⚠ 放在 evaluate/acg_eval **之后**：不改变任何既有读数，只是并列多打一档。
        # ⚠ 结果**单独存**（`all_sc`）而不是塞进 `all_res`：`all_res` 有 35 处消费点，
        #   改它的元数就得把所有解包点一起改，漏一处就是运行时 ValueError。
        if args.scale == 'fit':
            fitv = fit_scale(model, data, args, device)
            print(f'== 逐查询尺度：T̂ 拟合 ==  train n={fitv["n"]}  岭 λ={fitv["lam"]:.2f}  '
                  f'特征 {len(fitv["mu"])} 维（全无标签；val 未参与拟合）')
            all_sc.append((seed, scale_report(model, data, args, device, fitv)))
        all_res.append((seed, res, acg))

        if args.save:
            out = f'{args.out}_s{seed}'
            os.makedirs(out, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out, 'best.pt'))
            with open(os.path.join(out, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump({**vars(args), 'in_dim': data['feat'].size(1),
                                           'num_attrs': data['num_attrs']}, f,
                          ensure_ascii=False, indent=2)
            print(f'已保存模型到 {out}')

    if len(all_res) > 1:
        print(f'\n== 多 seed 汇总（{len(all_res)} seeds，mean ± std）==')
        # 配置回声：汇总段被单独摘出来传阅时，关键开关必须自带，否则无法判读
        print(f'  [配置] {os.path.basename(args.queries)}  loss={args.loss}'
              f'(pw_max={args.pw_max:.1f}'
              + (f',α={args.tv_alpha},β={args.tv_beta}' if args.loss == 'tversky' else '')
              + f')  mask_head={args.mask_head}  shape={args.shape}'
              + (f'  restrict_hop={args.restrict_hop}' if args.restrict_hop else '')
              + (f'  k_gran={args.k_gran}' if args.k_gran > 1 else '')
              + (f'  pos={args.pos_mode}/{args.pos_chan}'
                 + (f'+neg{args.pos_neg}' if args.pos_neg != 'off' else '')
                 if args.pos_mode != 'off' else '')
              + (f'  seed_sim={args.seed_sim}' if args.seed_sim != 'off' else '')
              + (f'  prompt_src={args.prompt_src}' if args.prompt_src != 'raw' else '')
              + (f'  neg_keep={args.neg_keep}' if args.neg_keep != 'mix' else '')
              + (f'  attr_only_p={args.attr_only_p:.2f}' if args.attr_only_p > 0 else '')
              + (f'  alt_neg_w={args.alt_neg_w:g}' if args.alt_neg_w > 0 else '')
              + (f'  anon_seed_p={args.anon_seed_p:g}' if args.anon_seed_p > 0 else '')
              + (f'  wd={args.weight_decay:g}' if args.weight_decay != 1e-4 else '')
              + f'  prior={args.prior}'
              + (f'(neg={args.prior_neg},α={args.prior_alpha})'
                 if args.prior != 'none' else '')
              + f'  prompt_cats={args.prompt_cats}'
              + (f'(neg_cover={args.neg_cover:g})'
                 if args.prompt_cats == 'taxonomy' else '')
              + f'  synth={args.synth_source}  seeds={args.seeds}')

        def ms(vals):
            return f'{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}'

        rows = [
            ('test +neg F1（macro，搜索主指标）', [r['f1_neg'] for _, r, _ in all_res]),
            ('增益 all', [r['gain_all'] for _, r, _ in all_res]),
            ('增益 overlap', [r['gain_overlap'] for _, r, _ in all_res]),
            ('增益 plain', [r['gain_plain'] for _, r, _ in all_res]),
            # ---- 阶段31：8 个提示类别的检索 F1（macro + plain 组）与相对 n 的配对增益 ----
            # 类别名：n=正节点（必备底座）、+a=正属性、-n=负节点、-a=负属性；
            # `full`/`n+a` 逐位等于旧的 +neg/base 两行；`rand` = 同数量**随机**负 token 的对照
            # （rand ≈ full ⇒ 增益只是"多两个 token"；full ≫ rand ⇒ 负点的身份有信息）。
            # `attr` = 纯属性诊断行（无节点 token），不参与早停/选择。
            ('test +neg P（full 提示）', [r['p_neg'] for _, r, _ in all_res]),
            ('test +neg R（full 提示）', [r['r_neg'] for _, r, _ in all_res]),
            ('test base P（n+a 提示）', [r['p_base'] for _, r, _ in all_res]),
            ('test base R（n+a 提示）', [r['r_base'] for _, r, _ in all_res]),
            *[row for c in PROMPT_CATS for row in (
                (f'类别[{c}] F1（macro）', [r['cat_all_%s' % c] for _, r, _ in all_res]),
                (f'类别[{c}] P（macro）', [r['p_cat_%s' % c] for _, r, _ in all_res]),
                (f'类别[{c}] R（macro）', [r['r_cat_%s' % c] for _, r, _ in all_res]),
            )],
            *[(f'类别[{c}] F1（plain 组）', [r['cat_pl_%s' % c] for _, r, _ in all_res])
              for c in PROMPT_CATS],
            ('类别[rand] F1（macro，随机负 token 对照）',
             [r['cat_all_rand'] for _, r, _ in all_res]),
            ('类别[rand] F1（plain 组，对照）',
             [r['cat_pl_rand'] for _, r, _ in all_res]),
            ('类别[attr] F1（macro，纯属性诊断）',
             [r['cat_all_attr'] for _, r, _ in all_res]),
        ]
        # 方案 4 的一等读出（只有 --k-gran>1 才有数）：K 个候选的分歧程度与选择器好坏。
        # Jaccard ≈1 / top-1 ≈ 0 都是"机制没生效"的信号，必须先看这几行再读其它指标。
        if args.k_gran > 1:
            rows = [
                ('K 粒度 候选两两 Jaccard（=1 → 候选相同）',
                 [r['k_jac'] for _, r, _ in all_res]),
                ('K 粒度 全 K 相同的查询比例', [r['k_same'] for _, r, _ in all_res]),
                ('K 粒度 top-1（质量头选）F1', [r['k_top1'] for _, r, _ in all_res]),
                ('K 粒度 best-of-K（oracle）F1', [r['k_best'] for _, r, _ in all_res]),
                ('K 粒度 选择器命中率', [r['k_hit'] for _, r, _ in all_res]),
            ] + rows

        def acg_rows(mode_key, tag):
            """CLARE/ProCom 口径（未过滤=全部种子输出）。"""
            def g(k):
                return [a[mode_key][k] for _, _, a in all_res]

            return [
                (f'ACG[{tag}] AvgF1（CLARE 口径·未过滤·主指标）', g('sym_f1')),
                (f'ACG[{tag}] AvgJaccard', g('sym_j')),
                (f'ACG[{tag}] ONMI', g('onmi')),
                (f'ACG[{tag}] detect%', g('detect_pct')),
                (f'ACG[{tag}] found', g('found')),
                (f'ACG[{tag}] 配对 F1', g('det_f1')),
                (f'ACG[{tag}] 重叠≥0.3 best-F1', g('expo_hi_f1_raw')),
                (f'ACG[{tag}] 低重叠 best-F1', g('expo_lo_f1_raw')),
            ]

        if not getattr(args, 'skip_detect', False):
            rows += acg_rows('节点+属性', 'n+a') + acg_rows('节点-only', 'node')
        # 逐查询尺度（阶段20）：同 run 配对，所以这三行不以 run 噪声为参照
        if all_sc:
            rows += [
                ('**逐查询尺度** [val] T=0 / T̂ / oracle-T',
                 [s['val']['f1_t0'] for _, s in all_sc]),
                ('**逐查询尺度** [val] T̂', [s['val']['f1_th'] for _, s in all_sc]),
                ('**逐查询尺度** [val] oracle-T（上界）',
                 [s['val']['f1_or'] for _, s in all_sc]),
                ('**逐查询尺度** [test] T=0', [s['test']['f1_t0'] for _, s in all_sc]),
                ('**逐查询尺度** [test] T̂', [s['test']['f1_th'] for _, s in all_sc]),
                ('**逐查询尺度** [test] oracle-T（上界）',
                 [s['test']['f1_or'] for _, s in all_sc]),
                ('**逐查询尺度** [test] T̂ 均值（档位；0=不切）',
                 [s['test']['T_mean'] for _, s in all_sc]),
                # ⚠ 标签写 **node**：这里的 ACG 种子是**单节点**提示（run.py 的 acg_eval 才有
                #   "节点+属性"一档），首版误标成 n+a，会与主表的 n+a 行混读。
                ('**逐查询尺度** [检测 node] T=0（同 run 配对）',
                 [s['acg_0']['sym'] if s.get('acg_0') else 0.0 for _, s in all_sc]),
                ('**逐查询尺度** [检测 node] T̂',
                 [s['acg_1']['sym'] if s.get('acg_1') else 0.0 for _, s in all_sc]),
            ]
        for name, vals in rows:
            print(f'  {name}: {ms(vals)}')
    else:
        print('\n（单 seed 运行；多 seed 汇总请传 --seeds 0 1 2 ...）')


if __name__ == '__main__':
    main()
