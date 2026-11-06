# -*- coding: utf-8 -*-
"""通用工具：指标、损失、随机种子。"""
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prf(pred: np.ndarray, target: np.ndarray):
    """二值向量的 P/R/F1。pred/target 必须是 bool 掩码数组（不接受节点 id 集合）。"""
    pred = np.asarray(pred)
    target = np.asarray(target)
    assert pred.dtype == bool and target.dtype == bool, (
        'prf 需要 bool 掩码数组（长度=节点数）；节点 id 集合请改用 jaccard()，'
        '或先转成掩码（如 np.isin(np.arange(n), sorted(s))）')
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    inter = float(np.logical_and(pred, target).sum())
    union = float(np.logical_or(pred, target).sum())
    return inter / union if union > 0 else 0.0


def dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """prob: (n,) sigmoid 概率；target: (n,) 0/1。"""
    prob = prob.reshape(-1)
    target = target.reshape(-1).float()
    inter = (prob * target).sum()
    return 1.0 - (2 * inter + eps) / (prob.sum() + target.sum() + eps)


def tversky_loss(prob: torch.Tensor, target: torch.Tensor,
                 alpha: float = 0.7, beta: float = 0.3,
                 eps: float = 1e-6) -> torch.Tensor:
    """Tversky 损失（概率版，soft）：1 − TP / (TP + α·FP + β·FN)。

    α 是**误检（FP）的权重**、β 是**漏检（FN）的权重**（Salehi 等的约定）：
      - α > β → 更恨误检 → 输出集合更紧（提高精度）；
      - α < β → 更恨漏检 → 输出更宽（提高召回）；
      - **α = β = 0.5 → 逐位等于 `dice_loss`**（走下面的短路，保证严格相等）。

    为什么要它：`compute_loss` 里的 `pos_weight` 在 facebook 上顶到 50，
    即漏检代价是误检的 50 倍，该损失的最优决策点是 p≈0.02 而非 0.5 —— 概率场被
    训练成"宁可多标"，实测操作点 P 0.180 / R 0.494（预测是 GT 的 2.74 倍），
    并且把场压平到 `oracle-size` 恒低于 0.5 阈值。α > β 是主动把操作点推回精度侧。
    """
    if alpha == beta:
        # α=β 时 Tversky 恒等于 Dice，直接转调以保证与 dice_loss 逐位一致
        # （否则两者只在 eps 的放置上差 ~1e-6）。这让 "--tv-alpha .5 --tv-beta .5"
        # 成为一个可断言的等价性对照。
        return dice_loss(prob, target, eps)
    prob = prob.reshape(-1)
    target = target.reshape(-1).float()
    tp = (prob * target).sum()
    fp = (prob * (1 - target)).sum()
    fn = ((1 - prob) * target).sum()
    return 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def ppr_score(seeds, src: np.ndarray, dst: np.ndarray, deg: np.ndarray, n: int,
              alpha: float = 0.15, iters: int = 20) -> np.ndarray:
    """多源个性化 PageRank 的**连续**概率向量（长度 n，和为 1）。

    幂迭代：p_{t+1} = alpha·e_S + (1-alpha)·W^T p_t，W[u,v] = 1/deg(u)。
    对称边表（无向边存两次、无自环）下 `bincount(dst, weights=p[src]/deg[src])`
    正是 W^T p 那一项，所以这里只按 src 取度数、不折半。

    与 `synth.ppr_cone` 的关系：那是同一递推的 **top-k 集合**版本（合成圈生成用），
    这里要整条向量才能做逐节点软先验与"正锥 − λ·负锥"。两者在单源上取 top-k 的概率
    值逐位相同（已由 temp/verify_cone_diag.py 钉死），所以这里**刻意不改 synth.py**，
    避免动到已验证的合成管线行为。

    用途（见 RUN-LIST 阶段8）：方案 2/3 的查询局部先验 —— 诊断显示
    正点锥的质心系统性偏向混淆社区 B（同号比例仅 32.5%），而"正锥 − λ·负锥"把
    目标 A 的 Dice 从 0.427 抬到 0.461（λ≈0.5~1，倒 U 型）。
    """
    e = np.zeros(n, dtype=np.float64)
    idx = list(seeds)
    if not idx:
        return e
    e[idx] = 1.0 / len(idx)
    p = e.copy()
    w = 1.0 / np.maximum(deg, 1.0)
    for _ in range(iters):
        contrib = p[src] * w[src]
        p_new = alpha * e + (1.0 - alpha) * np.bincount(dst, weights=contrib,
                                                        minlength=n)
        converged = np.abs(p_new - p).sum() < 1e-10
        p = p_new
        if converged:
            break
    return p


def conductance(edge_index: np.ndarray, mask: np.ndarray, n: int) -> float:
    """edge_index: (2,E)（按 sampleQuery._load_graph 的约定：无向边存两次、无自环）；
    mask: (n,) bool。返回标准割传导率 cut(S) / min(vol(S), vol(V\\S))，取值 [0,1]。

    两个容易踩的点（都在这函数里踩过，改动前请连同 temp/verify_numpy_paths.py 一起看）：
      1) cut 只能统计**恰好一个端点**在 S 内的边。写成 `~(mask[src] & mask[dst])` 会把
         "两端都在 S 外"的边也算进来，该式等于 (m - e_S) / min(vol_S, vol_T)，对小集合
         退化成 ≈ m / vol_S，量级可到上千（日志里 120.6 / 690.2 就是它），不是传导率。
      2) 每条无向边存了两次 → 有向跨越计数是真实 cut 的两倍，要折半；而度数只能用 src
         统计（src 已含双向，正好等于真实度）。若沿用 concat([src, dst]) 算度数，度数会
         再翻一倍，两者相抵 → 结果恰好偏小一半。
    """
    src, dst = edge_index
    cut = 0.5 * float((mask[src] != mask[dst]).sum())       # 每条无向边被记两次 → 折半
    deg = np.bincount(src, minlength=n).astype(np.float64)  # src 已含双向 → 真实度
    vol_s = float(deg[mask].sum())
    vol_t = float(deg.sum()) - vol_s
    denom = min(vol_s, vol_t)
    return cut / denom if denom > 0 else 1.0


def stability_score(logits: np.ndarray, delta: float = 2.0) -> float:
    """SAM 式稳定性：logits 在 {>0, >delta} 两个阈值下二值化的 IoU（无真值质量分）。"""
    return iou(logits > 0.0, logits > delta)


# ==================== 直接读取 sampleQuery.py 生成的查询文件 ====================
def load_queries(path):
    """读 _queries.pt，返回 (queries, graph)。queries 为扁平的查询列表。

    ---- 格式 v2（阶段47）：全图信息只存顶层一份 ----
    旧格式把 `graph` 挂在每条查询下；新格式是 `{'graph': g, 'query_batchs': [...]}`。
    两种都读，并把图**以同一个对象引用重新挂回**每条查询 —— 下游大量代码直接用
    `q['graph']`（QDGNN / cam_split / synth），这样它们不用改，内存里也只有一份。
    """
    d = torch.load(path, map_location='cpu')
    qs = [q for batch in d['query_batchs'] for q in batch]
    g = d.get('graph')
    if g is None and qs:
        g = qs[0].get('graph')          # 旧格式：图挂在查询上
    if g is not None:
        # 阶段47b：新文件不再写 `attributes`（与 node_feat 前 num_attributes 列重复），
        # 这里按需切片重建，下游（run.acg_eval / synth）拿到的仍是同一份数据。
        # ⚠ 压缩属性的文件（有 attr_dim）前 num_attributes 列**不是**属性而是嵌入，不能切。
        if 'attributes' not in g and 'node_feat' in g and 'attr_dim' not in g:
            _m = int(g['num_attributes'])
            g['attributes'] = g['node_feat'][:, :_m]
        for q in qs:
            q['graph'] = g              # 共享引用，不复制
    return qs, g


def make_prompt(q):
    """查询 -> 提示 token 列表 [{kind,id,pol}]。kind 0=节点 1=属性；pol 0=正 1=负。"""
    out = [{'kind': 0, 'id': int(v), 'pol': 0} for v in q['query']['nodes'].tolist()]
    out += [{'kind': 1, 'id': int(a), 'pol': 0} for a in q['query']['attrs'].tolist()]
    out += [{'kind': 0, 'id': int(v), 'pol': 1} for v in q['query']['neg_nodes'].tolist()]
    out += [{'kind': 1, 'id': int(a), 'pol': 1} for a in q['query']['neg_attrs'].tolist()]
    return out


def make_prompt_noneg(q):
    """只取正提示（不含负），评估 base 变体用。"""
    out = [{'kind': 0, 'id': int(v), 'pol': 0} for v in q['query']['nodes'].tolist()]
    out += [{'kind': 1, 'id': int(a), 'pol': 0} for a in q['query']['attrs'].tolist()]
    return out


def make_prompt_attr(q):
    """**纯属性提示**（诊断变体）：去掉全部节点 token，只留属性 token（正 + 负）。

    动机（output67/68）：ACG 里 n+a 只比 node-only 高 +0.006（0.393 vs 0.387，found 同为 7/11）、
    检索表里这一行实测 F1 0.049（掩码均值 3.5 个节点 vs GT 53.6），而 base 0.314 —— 检索
    分数几乎全部来自种子点。属性取自 `community_attributes`（**社区** top-3、共享属性优先，
    由 sampleQuery* 生成），即"社区级语义画像"，是唯一能迁移到"训练集里没有的社区"的输入；
    而种子点本身就落在目标社区内，模型完全可以背出"这个点属于哪个训练社区"（塌缩诊断 85%
    的最佳匹配落在训练社区）。所以这一行要单独量：`base − attr` = 种子点的贡献，
    `attr` 自己 = 只有语义画像、没有"点在哪"时的水平。

    ⚠ 这**不是**分布外提示：`train()` 的 `--attr-aux` 辅助任务（默认 0.1）每轮都拿
    **同样构造**的纯属性提示（`make_prompt` 里 `kind == 1` 的过滤）算损失 —— 所以这一行
    读的是"属性通道学到什么程度"，不是"没见过的形态"。定位与 `attr_only_p`（阶段23，
    把同一条提示放进主任务）配套。
    ⚠ 无正点 ⇒ `prompt_ball` 返回 None（没有正点就不张球）⇒ `--restrict-hop>0` 时该行
    **不做球限制**、直接在整图上评。这是刻意的（属性提示没有可张球的中心）；当前配置
    `--restrict-hop 0` 下两种写法逐位等价。
    """
    out = [{'kind': 1, 'id': int(a), 'pol': 0} for a in q['query']['attrs'].tolist()]
    out += [{'kind': 1, 'id': int(a), 'pol': 1} for a in q['query']['neg_attrs'].tolist()]
    return out


# ==================== 提示类别（8 种，阶段31）====================
# 规则：**正节点必备**；在它之上叠加 {正属性, 负节点, 负属性} 的 7 个非空子集。
# 类别名即组成：n=正节点、+a=正属性、-n=负节点、-a=负属性；'full' = 四者全带。
CAT_COMPS = (
    ('n',     (False, False, False)),
    ('n+a',   (True,  False, False)),
    ('n-n',   (False, True,  False)),
    ('n-a',   (False, False, True)),
    ('n+a-n', (True,  True,  False)),
    ('n+a-a', (True,  False, True)),
    ('n-n-a', (False, True,  True)),
    ('full',  (True,  True,  True)),
)
CAT_COMP = dict(CAT_COMPS)
PROMPT_CATS = tuple(c for c, _ in CAT_COMPS)
NEG_CATS = tuple(c for c, (a, n, t) in CAT_COMPS if n or t)          # 6 个带负类别
POSONLY_CATS = tuple(c for c in PROMPT_CATS if c not in NEG_CATS)    # 2 个无负类别


def make_prompt_cat(q, cat):
    """按类别构造提示（正节点必备）。

    ⚠ 与既有构造器的对应关系（守卫 temp/check_cats.py 逐查询断言）：
      make_prompt_cat(q, 'full') == make_prompt(q)          （正点+正属性+负点+负属性）
      make_prompt_cat(q, 'n+a')  == make_prompt_noneg(q)    （只有正点+正属性）
    token 顺序与 make_prompt 完全一致（节点→属性→负节点→负属性），所以旧日志的
    `+neg` / `base` 两列分别等于本表的 `full` / `n+a` 两行，历史数字可直接对照。
    """
    a, n, t = CAT_COMP[cat]
    out = [{'kind': 0, 'id': int(v), 'pol': 0} for v in q['query']['nodes'].tolist()]
    if a:
        out += [{'kind': 1, 'id': int(v), 'pol': 0}
                for v in q['query']['attrs'].tolist()]
    if n:
        out += [{'kind': 0, 'id': int(v), 'pol': 1}
                for v in q['query']['neg_nodes'].tolist()]
    if t:
        out += [{'kind': 1, 'id': int(v), 'pol': 1}
                for v in q['query']['neg_attrs'].tolist()]
    return out


def has_pos_node(plist):
    """类别不变式：正节点必备（至少一个 kind==0 & pol==0 的 token）。"""
    return any(p['kind'] == 0 and p['pol'] == 0 for p in plist)


def make_prompt_rand(q, rng, n_nodes, n_attrs):
    """对照类别 `rand`：正点 + 正属性 + **同数量的随机**"负点/负属性"（同 pol、随机 id）。

    目的：把"+0.01 来自负信息"与"+0.01 只是多了两个 token"分开。本类别与 `full` 的
    token 数量、种类、极性完全相同，只有 id 是随机的（排除该查询自己的正点与真负点）。
    判据：`rand ≈ full > n+a` ⇒ 增益来自"多 token"；`full` 明显高于 `rand` ⇒ 负点的
    **身份**真的有信息。抽签用调用方给的 rng（评测侧按 seed 固定，各诊断块共用一份缓存）。
    """
    kn = len(q['query']['neg_nodes'].tolist())
    ka = len(q['query']['neg_attrs'].tolist())
    used_n = set(int(v) for v in q['query']['nodes'].tolist()) \
        | set(int(v) for v in q['query']['neg_nodes'].tolist())
    used_a = set(int(v) for v in q['query']['attrs'].tolist()) \
        | set(int(v) for v in q['query']['neg_attrs'].tolist())
    out = make_prompt_cat(q, 'n+a')
    if kn:
        pool = [v for v in range(n_nodes) if v not in used_n]
        if pool:
            out += [{'kind': 0, 'id': int(v), 'pol': 1}
                    for v in rng.sample(pool, min(kn, len(pool)))]
    if ka:
        pool = [v for v in range(n_attrs) if v not in used_a]
        if pool:
            out += [{'kind': 1, 'id': int(v), 'pol': 1}
                    for v in rng.sample(pool, min(ka, len(pool)))]
    return out


def cat_weights(neg_cover=2.0 / 3.0):
    """类别采样权重：无负类别共 (1-neg_cover)，带负类别共 neg_cover。

    默认 neg_cover=2/3 ⇒ 无负两类各 1/6、带负六类各 1/9。为什么默认不给 1：
    output59（负点覆盖率 1.00×199 轮）的代价是**无负提示的 base 通路被打坏**
    （掩码均值 150→423、oracle-size 余量被吃、检测 −0.025）——所以默认留 1/3 的无负
    形态给 base 通路。要复现 59 那种全覆盖，传 `--neg-cover 1`。
    """
    free = (1.0 - neg_cover) / len(POSONLY_CATS)
    withn = neg_cover / len(NEG_CATS)
    return [withn if c in NEG_CATS else free for c in PROMPT_CATS]


def error_driven_prompts(comm_set, pred, rng, max_add=2):
    """SAM 错误驱动交互点：漏检（真值内未预测到）→ 正点；误报（预测但真值外）→ 负点。"""
    fn = [int(v) for v in comm_set if not pred[v]]
    fp = [int(v) for v in np.where(pred)[0] if int(v) not in comm_set]
    inc = []
    for _ in range(rng.randint(0, max_add)):
        if fn and (not fp or rng.random() < 0.5):
            inc.append({'kind': 0, 'id': int(rng.choice(fn)), 'pol': 0})
        elif fp:
            inc.append({'kind': 0, 'id': int(rng.choice(fp)), 'pol': 1})
        else:
            break
    return inc


def split_queries(qs, seed, ratio=(0.6, 0.2, 0.2)):
    """随机打散并按比例划分 train/val/test（不打乱原列表，多 seed 可重复划分）。"""
    rng = random.Random(seed)
    qs = list(qs)
    rng.shuffle(qs)
    n = len(qs)
    n_tr, n_va = int(n * ratio[0]), int(n * ratio[1])
    return {'train': qs[:n_tr], 'val': qs[n_tr:n_tr + n_va], 'test': qs[n_tr + n_va:]}


def degree_features(edge_index, n, bins=16):
    """log-degree 等频分桶 one-hot，作为与属性无关的结构侧输入。

    （旧版结构特征，仅用于兼容没有 node_feat 字段的旧 .pt；新流程用
    struct_features / build_node_feat。）
    """
    deg = np.bincount(edge_index[0].numpy(), minlength=n).astype(np.float64)
    logd = np.log1p(deg)
    cuts = np.quantile(logd, np.linspace(0, 1, bins + 1)[1:-1])
    b = np.digitize(logd, cuts)
    return torch.from_numpy(np.eye(bins, dtype=np.float32)[b])


def struct_features(edge_index, n):
    """结构特征：[deg(u), min/max/mean/std(邻居度)]，逐列 z-score 标准化（5 维）。

    刻意与 KDD'22 CLARE / KDD'24 ProCom 的 `feature_augmentation` 逐项一致——
    即"现有社区检测模型同样考虑的结构信息"——这样论文里可以说
    "本方法与基线使用同一套结构输入"，而不是各用各的。
    （基线侧的同名实现在 cam_split.py:aug_features，公式相同。）

    edge_index 约定为无向边存两次（见 sampleQuery._load_graph），所以
    `bincount(edge_index[0])` 正好等于真实度。
    """
    src = np.asarray(edge_index[0].numpy() if hasattr(edge_index[0], 'numpy')
                     else edge_index[0]).astype(np.int64)
    dst = np.asarray(edge_index[1].numpy() if hasattr(edge_index[1], 'numpy')
                     else edge_index[1]).astype(np.int64)
    deg = np.bincount(src, minlength=n).astype(np.float64)
    nb = deg[dst]                                  # 每条有向边：目标端的邻居度

    cnt = np.bincount(src, minlength=n).astype(np.float64)
    sm = np.bincount(src, weights=nb, minlength=n)
    sq = np.bincount(src, weights=nb * nb, minlength=n)
    mn = np.full(n, np.inf)
    mx = np.zeros(n)
    np.minimum.at(mn, src, nb)
    np.maximum.at(mx, src, nb)

    has = cnt > 0
    mean = np.divide(sm, np.maximum(cnt, 1.0))
    var = np.divide(sq, np.maximum(cnt, 1.0)) - mean * mean
    f = np.zeros((n, 5), dtype=np.float64)
    f[:, 0] = deg
    f[has, 1] = mn[has]
    f[has, 2] = mx[has]
    f[has, 3] = mean[has]
    f[has, 4] = np.sqrt(np.maximum(var[has], 0.0))
    f = (f - f.mean(0, keepdims=True)) / (f.std(0, keepdims=True) + 1e-9)
    return torch.from_numpy(f.astype(np.float32))


def build_node_feat(attributes, edge_index, n):
    """节点特征 = 原属性 ⊕ 结构特征（唯一构造入口）。

    返回 (node_feat, struct_feat)：
      node_feat   (n, num_attrs+5) 送入 GNN 的特征
      struct_feat (n, 5)           单独存一份，便于诊断与复用

    放在这里（而不是 run.py）是为了让**查询生成文件**把特征写进 .pt：
    特征属于数据，不属于训练脚本；这样任何读取该 .pt 的下游（基线、分析脚本、
    不同版本的 run.py）拿到的都是同一份特征。
    """
    x = torch.as_tensor(np.asarray(attributes), dtype=torch.float32)
    sf = struct_features(edge_index, n)
    return torch.cat([x, sf], dim=1), sf
