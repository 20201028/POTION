# -*- coding: utf-8 -*-
"""POTION 模型三件套：

  GraphEncoder  提示无关图编码器：3 层均值聚合、按边分块，每图只前向一次。
  PromptEncoder 提示 -> token（基础嵌入 ⊕ 极性 ⊕ 来源）；空提示 -> NULL token。
  MaskDecoder   3 层 two-way transformer；K 个粒度 token（0=核心, 1=社区）；
                动态 MLP 头 M_k = <H, e_k>；质量头预测掩码 IoU。
"""
import torch
import torch.nn as nn


class PromptEncoder(nn.Module):
    def __init__(self, d, num_attrs, feat_dim, prompt_src='raw'):
        super().__init__()
        self.d = d
        if prompt_src not in ('raw', 'H'):
            raise ValueError('prompt_src 只能是 raw/H，得到 %r' % (prompt_src,))
        self.prompt_src = prompt_src
        # 提示编码器只接触原始输入（与图编码器完全解耦，对齐 SAM 的模块分工）：
        # 节点提示 = 原始特征（词袋⊕度分桶）投影；属性提示 = 属性嵌入查表。
        #
        # ---- 阶段38 `--prompt-src H`：节点 token 改吃**图编码后的表示** ----
        # 这条正是上面那句"模块分工"的对照实验：token = node_proj_h(H[v]) 而不是
        # node_proj(feat[v])，等于把提示侧并进图编码器（梯度会从提示 token 回流进编码器）。
        # 与 --seed-sim 的区别：那条是**加一列特征**（拼接、detach、不进编码器）；这条换的是
        # **提示 token 的来源**，是一条完全不同的梯度路径。属性 token 两模式都不变
        # （属性没有"图表示"这种东西，仍是 attr_emb 查表）。
        # ⚠ raw（默认）时**不建** node_proj_h ⇒ state_dict 键名与旧模型逐格一致，旧 .pt 可加载。
        if prompt_src == 'H':
            self.node_proj_h = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        else:
            self.node_proj = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU(), nn.Linear(d, d))
        self.attr_emb = nn.Embedding(num_attrs, d)
        self.pol_emb = nn.Embedding(2, d)   # 0=+, 1=-
        self.src_emb = nn.Embedding(2, d)   # 0=节点, 1=属性
        self.null = nn.Parameter(torch.zeros(d))

    def forward(self, prompt_lists, x):
        """prompt_lists: List[List[prompt dict]]；x: 节点 token 的来源张量 ——
        `prompt_src='raw'`（默认）时是 (n, feat_dim) 原始输入特征，`'H'` 时是 (n, d) 图编码
        表示。由调用方（run.py 的 `decode`）按 `--prompt-src` 把对的张量递进来。
        返回 (B,T,d) 提示 token 序列与 (B,T) pad 掩码（True=填充位）。"""
        dev = x.device
        seqs = []
        for pl in prompt_lists:
            if len(pl) == 0:                       # 无提示 -> NULL token
                seqs.append(self.null.unsqueeze(0))
                continue
            idx = torch.tensor([p['id'] for p in pl], dtype=torch.long, device=dev)
            pol = torch.tensor([p['pol'] for p in pl], dtype=torch.long, device=dev)
            src = torch.tensor([p['kind'] for p in pl], dtype=torch.long, device=dev)
            node_pos = [i for i, p in enumerate(pl) if p['kind'] == 0]
            attr_pos = [i for i, p in enumerate(pl) if p['kind'] == 1]
            base = torch.empty(len(pl), self.d, device=dev)
            if node_pos:
                npos = torch.tensor(node_pos, dtype=torch.long, device=dev)
                proj = self.node_proj if self.prompt_src == 'raw' else self.node_proj_h
                base[npos] = proj(x[idx[npos]])
            if attr_pos:
                apos = torch.tensor(attr_pos, dtype=torch.long, device=dev)
                base[apos] = self.attr_emb(idx[apos])
            seqs.append(base + self.pol_emb(pol) + self.src_emb(src))
        tmax = max(s.size(0) for s in seqs)
        toks = torch.zeros(len(seqs), tmax, self.d, device=dev)
        mask = torch.ones(len(seqs), tmax, dtype=torch.bool, device=dev)
        for i, s in enumerate(seqs):
            toks[i, :s.size(0)] = s
            mask[i, :s.size(0)] = False
        return toks, mask


class TwoWayLayer(nn.Module):
    """一层双向交互：token 自注意力 -> token<->H 双向交叉注意力 -> 双侧 FFN。"""

    def __init__(self, d, heads=4):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.t2h = nn.MultiheadAttention(d, heads, batch_first=True)  # token 读节点
        self.h2t = nn.MultiheadAttention(d, heads, batch_first=True)  # 节点读 token
        self.f_t = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.f_h = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(8)])

    def forward(self, tok, H, pad, node_pad=None):
        # 1) token 自注意力（pad 位被 key_padding_mask 屏蔽）
        t = tok + self.sa(self.ln[0](tok), self.ln[1](tok), self.ln[2](tok),
                          key_padding_mask=pad, need_weights=False)[0]
        # 2) token <- 节点（提示/粒度 token 从图里读取证据）
        t = t + self.t2h(self.ln[3](t), self.ln[4](H), self.ln[5](H),
                         key_padding_mask=node_pad, need_weights=False)[0]
        # 3) 节点 <- token（SAM 的 two-way：节点嵌入也被提示更新）
        h = H + self.h2t(self.ln[6](H), self.ln[7](t), self.ln[7](t),
                         key_padding_mask=pad, need_weights=False)[0]
        # 4) FFN
        t = t + self.f_t(self.ln[3](t))   # 复用 ln[3]
        h = h + self.f_h(self.ln[4](h))   # 复用 ln[4]
        return t, h


class MaskDecoder(nn.Module):
    def __init__(self, d=128, k_gran=1, layers=2, heads=4, mask_head='dot',
                 pos_mode='off', pos_dim=0, attr_mode='off', attr_init=1.0):
        super().__init__()
        self.k = k_gran
        self.mask_head = mask_head
        # ---- 相对位置通道（阶段12）----
        # pos_mode='off' 时**不建任何参数**（state_dict 键名与旧模型逐位一致，旧 .pt 仍可加载）。
        if pos_mode not in ('off', 'feat', 'head'):
            raise ValueError('pos_mode 只能是 off/feat/head，得到 %r' % (pos_mode,))
        if pos_mode == 'off':
            self.pos_dim = 0
        else:
            if pos_dim <= 0:
                raise ValueError('pos_mode=%s 需要 pos_dim>0，得到 %r'
                                 % (pos_mode, pos_dim))
            self.pos_dim = int(pos_dim)
        self.pos_mode = pos_mode
        self.queries = nn.Parameter(torch.randn(k_gran, d) * 0.02)  # 单粒度（社区）
        self.layers = nn.ModuleList([TwoWayLayer(d, heads) for _ in range(layers)])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            for _ in range(k_gran)])
        self.ln_e = nn.LayerNorm(d)
        self.ln_h = nn.LayerNorm(d)
        self.quality = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # 查询局部先验的两个信任度：[正锥, 负锥]。**初值全 0** → 训练起点与"无先验"
        # 逐位相同（与 --mask-head mlp 的零初始化同一手法），所以任何差异都只能来自
        # 学到的部分；且 gamma 的收敛值本身就是"先验有没有被用上"的可报告读数。
        # 为什么加在解码器而不是编码器：编码器必须保持提示无关（CAM 对齐 SAM 的核心），
        # 且 ACG 用 3959 个种子共享一次编码——先验依赖提示，拼进 feat 会强制每查询重编码。
        self.prior_w = nn.Parameter(torch.zeros(2))
        # ---- 属性通道（--attr-mode sim，阶段45）----
        # 动机：cora 上 `attr`（纯属性提示）与 `full` 的 F1 与掩码**完全相同**（seed 0/2
        # 都是 0.360、seed 3 都是 0.001），说明属性 token 只经由 two-way transformer
        # 间接影响 query，实际从未进入"算掩码的那一步"（logits = ⟨h_v,e⟩/√d）。
        # 而 cora/facebook 的社区/类别恰恰是词袋可分的（`attr` 行仅 0.047/0.024）。
        # 这里把"节点属性向量 × 提示属性向量"的余弦相似度**直接加到 logits**，
        # 由单个可学习标量 attr_w 控制强度；字段本身由 run.attr_field 计算（纯数据、
        # 无参数、无标签），模型侧只做加权注入。
        # off ⇒ **不建任何参数**，state_dict 键名与旧模型逐格一致（旧 .pt 仍可加载）；
        # attr_init=0.0 时起点与 off 逐位相同（同 prior_w / pos_* 的零初始化纪律）。
        self.attr_mode = attr_mode
        if attr_mode not in ('off', 'sim'):
            raise ValueError('attr_mode 只能是 off/sim，得到 %r' % (attr_mode,))
        if attr_mode != 'off':
            self.attr_w = nn.Parameter(torch.tensor(float(attr_init)))
        if self.pos_dim > 0:
            # 相对位置通道：把 r(v|种子) 接进"算掩码的那一步"。动机（阶段12）：掩码
            # ⟨h_v, e⟩/√d 里没有任何一项依赖 **(节点, 提示) 的关系**，所以模型只能检索
            # 训练时记住的社区（98% 查询的最佳匹配不是目标），而种子的邻域里概率≈0
            # （--restrict-hop 一限制，plain 20 条里 15 条直接变空）。SAM 的对应物是把
            # 位置加进**解码器操作的那批特征**（`src = src + image_pe`），不是加在最后
            # 的分割图上 —— 所以这里同样加在 two-way 之前，而不是当输出偏置。
            # 两条通道的末层都**零初始化** → `--pos-mode feat/head` 的第一步前向与
            # `off` **逐位相同**，任何差异都只能来自学到的部分（与 prior_w / mask_head
            # mlp 同一手法，冒烟测试里逐位对拍）。
            if pos_mode == 'feat':
                # logits 侧仍是点积；位置只改节点表示（Hb += W·r），交叉注意力与掩码
                # 头看到的是"带位置"的节点特征。
                self.pos_proj = nn.Linear(self.pos_dim, d)
                nn.init.zeros_(self.pos_proj.weight)
                nn.init.zeros_(self.pos_proj.bias)
            else:
                # 逐 (节点, 候选粒度) 的 [点积, r] 过一个小 MLP 再残差加回 logits：让头能
                # 学"r 小就信点积、r 大就压掉"这种**节点依赖的陡决策** —— 这正是方案 3 的
                # 加性标量场 γ·w 表达不了的东西（阶段8/9 的否决理由）。
                self.pos_head = nn.Sequential(
                    nn.Linear(self.pos_dim + 1, d), nn.GELU(), nn.Linear(d, 1))
                nn.init.zeros_(self.pos_head[-1].weight)
                nn.init.zeros_(self.pos_head[-1].bias)
        if mask_head == 'mlp':
            # 掩码头 = 点积 + 残差 MLP，输入是逐节点成对特征 [h⊙e, |h−e|]。
            # 末层**零初始化** → 训练开始时 logits 逐位等于点积头，所以与 --mask-head dot
            # 的对照是"从同一点出发"，任何差异都只能来自学到的部分（不是初始化造成的）。
            # 动机：点积是"把节点嵌入投影到一个方向再取阈值"，表达力弱；而诊断显示
            # 概率场在候选区域内部排不出谁属于社区（oracle-size 与原样相等），
            # 需要能表达"在哪些维度上匹配"的成对特征。
            self.mask_mlp = nn.Sequential(
                nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
            nn.init.zeros_(self.mask_mlp[-1].weight)
            nn.init.zeros_(self.mask_mlp[-1].bias)

    def forward(self, ptok, pmask, H, prior=None, pos=None, attr_bias=None,
                node_pad=None):
        """ptok: (B,T,d) 提示 token；pmask: (B,T) True=pad；H: (n,d)。
        prior: (B,2,n) 或 None —— [:,0]=正点 PPR 锥、[:,1]=负点 PPR 锥（都已按最大值归一化）。
        pos: (B,n,pos_dim) 或 None —— 逐节点**相对提示**的位置（阶段12，由 run.query_pos
        从提示正点导出）。pos_mode='off' 或 pos=None 时整条通道不参与，逐位等价旧行为。
        attr_bias: (B,n) 或 None —— 逐节点的**属性相似度场**（阶段45，由 run.attr_field
        从提示的属性 token 导出）。attr_mode='off' 或 None 时整条通道不参与。
        返回 logits (B,K,n)（未过 sigmoid）与 quality (B,K)（预测 IoU）。"""
        B = ptok.size(0)
        dev = H.device
        Hb = H.unsqueeze(0).expand(B, -1, -1) if H.dim() == 2 else H
        if self.pos_dim and pos is not None and self.pos_mode == 'feat':
            Hb = Hb + self.pos_proj(pos.to(Hb.dtype))
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        tok = torch.cat([q, ptok], dim=1)
        pad = torch.cat([torch.zeros(B, self.k, dtype=torch.bool, device=dev), pmask], dim=1)
        for layer in self.layers:
            tok, Hb = layer(tok, Hb, pad, node_pad)
        e = torch.stack([mlp(tok[:, k]) for k, mlp in enumerate(self.heads)], dim=1)  # (B,K,d)
        e = self.ln_e(e)
        h = self.ln_h(Hb)
        logits = torch.einsum('bnd,bkd->bkn', h, e) / (e.size(-1) ** 0.5)
        if prior is not None:
            # logits += γ·w_pos − λ·w_neg（prior_w 初值 0 → 逐位等价于无先验）
            # prior_w 是 1 维 (2,)，prior 是 (B,2,n)：'c,bcn->bn' 对第 1 维加权求和，
            # 再 unsqueeze 成 (B,1,n) 与 (B,K,n) 广播。
            logits = logits + torch.einsum('c,bcn->bn', self.prior_w, prior).unsqueeze(1)
        if self.mask_head == 'mlp':
            hh = h.unsqueeze(1)                                   # (B,1,n,d)
            ee = e.unsqueeze(2)                                   # (B,K,1,d)
            z = torch.cat([hh * ee, (hh - ee).abs()], dim=-1)     # (B,K,n,2d)
            logits = logits + self.mask_mlp(z).squeeze(-1)
        if self.pos_dim and pos is not None and self.pos_mode == 'head':
            # 逐 (b,k,节点) 的 [点积, r]：(B,1,n,d)→(B,K,n,d) 与 (B,K,n,1) 拼接后过 MLP。
            # 末层零初始化 → 起始时这一项恒为 0，故与 off 逐位相同。
            zr = torch.cat([logits.unsqueeze(-1),
                            pos.to(logits.dtype).unsqueeze(1).expand(
                                -1, logits.size(1), -1, -1)], dim=-1)
            logits = logits + self.pos_head(zr).squeeze(-1)
        if self.attr_mode != 'off' and attr_bias is not None:
            # (B,n) 余弦场 -> (B,1,n)，对全部 K 个槽位广播；attr_w 可学习。
            # 放在 pos_head 之后：它是"证据场"，不是对已有 logits 的再校准。
            logits = logits + self.attr_w * attr_bias.to(logits.dtype).unsqueeze(1)
        quality = torch.sigmoid(self.quality(e)).squeeze(-1)
        return logits, quality


class CAM(nn.Module):
    def __init__(self, in_dim, num_attrs, d=128, k_gran=1, layers=2,
                 mask_head='dot', pos_mode='off', pos_dim=0, prompt_src='raw',
                 attr_mode='off', attr_init=1.0, frozen_H=None):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, d, layers)
        # --frozen-h：编码器冻结，直接使用预计算的 H
        self.frozen_H = frozen_H
        if frozen_H is not None:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.prompt_enc = PromptEncoder(d, num_attrs, feat_dim=in_dim,
                                       prompt_src=prompt_src)
        self.decoder = MaskDecoder(d, k_gran, layers, mask_head=mask_head,
                                   pos_mode=pos_mode, pos_dim=pos_dim,
                                   attr_mode=attr_mode, attr_init=attr_init)

    def encode(self, feat, edge_index):
        # 冻结模式下训练循环也走这里（train() 不经过 forward）
        return self.frozen_H if self.frozen_H is not None else self.encoder(feat, edge_index)

    def forward(self, prompt_lists, feat, edge_index, prior=None, pos=None,
                attr_bias=None):
        H = self.encode(feat, edge_index)
        x = H if self.prompt_enc.prompt_src == 'H' else feat
        ptok, pmask = self.prompt_enc(prompt_lists, x)
        return self.decoder(ptok, pmask, H, prior, pos, attr_bias=attr_bias)


# ==================== 分块 SAGE 编码器（联合训练用） ====================
from .chunked_encoder import ChunkedSAGE


class GraphEncoder(nn.Module):
    """与提示无关的编码器：3 层均值聚合，按边分块做全图前向。

    显存 O(chunk*d) 而不是 O(E*d)，因此可以在 6.8 万节点 / 116 万边的图上
    与解码器一起反向传播（联合训练）。
    """

    def __init__(self, in_dim, d=128, layers=3, chunk=2_000_000):
        super().__init__()
        self.inner = ChunkedSAGE(in_dim, d, layers)
        self.chunk = int(chunk)

    def forward(self, feat, edge_index):
        return self.inner(feat, edge_index, chunk=self.chunk)
