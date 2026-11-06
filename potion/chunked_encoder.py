import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _ChunkedAgg(torch.autograd.Function):
    """无参数的块状均值聚合，前向/反向都只占 O(chunk*d)。

    聚合本身没有参数，所以反向只需要把 grad_out[dst] 散射回 src，
    完全不必保存 h[src]——这是 6 千万边图上能训练的关键。
    """

    @staticmethod
    def forward(ctx, h, src, dst, n, chunk):
        ctx.save_for_backward(src, dst)
        ctx.n, ctx.chunk, ctx.d = int(n), int(chunk), h.size(1)
        out = torch.zeros(ctx.n, ctx.d, device=h.device, dtype=h.dtype)
        for s in range(0, src.numel(), ctx.chunk):
            out.index_add_(0, dst[s:s + ctx.chunk], h[src[s:s + ctx.chunk]])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        src, dst = ctx.saved_tensors
        grad_h = torch.zeros(ctx.n, ctx.d, device=grad_out.device, dtype=grad_out.dtype)
        for s in range(0, src.numel(), ctx.chunk):
            grad_h.index_add_(0, src[s:s + ctx.chunk], grad_out[dst[s:s + ctx.chunk]])
        return grad_h, None, None, None, None


class ChunkedSAGE(nn.Module):
    """Mean-aggregator SAGE, full-graph forward in exact edge chunks.

    Peak memory is O(chunk * d) for both forward and backward, so a
    123.7M-edge graph trains on a 16 GB card with chunk <= 2e5.
    """

    def __init__(self, in_dim, d=128, layers=3):
        super().__init__()
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            self.lins.append(nn.Linear(in_dim if i == 0 else d, d))
            self.norms.append(nn.LayerNorm(d))

    def forward(self, x, edge_index, chunk=200_000, ckpt='auto'):
        h = x
        src_all, dst_all = edge_index[0], edge_index[1]
        if src_all.dtype != torch.long:
            src_all, dst_all = src_all.long(), dst_all.long()   # index_add 要求 long
        n = h.size(0)
        deg = torch.bincount(dst_all, minlength=n).clamp(min=1).unsqueeze(1).to(h.dtype)
        _big = n * h.size(1) * 4 > (1 << 27)          # >128 MB 视为大图
        use_ckpt = (ckpt != 'off' and torch.is_grad_enabled()
                    and bool(self.lins[0].weight.requires_grad)
                    and (ckpt == 'on' or _big))
        if not getattr(self, '_ckpt_logged', False):
            print(f'[chunked] ckpt={use_ckpt} chunk={chunk} n={n} d={h.size(1)}', flush=True)
            self._ckpt_logged = True
        for lin, ln in zip(self.lins, self.norms):
            def layer(h_, lin=lin, ln=ln):
                out = _ChunkedAgg.apply(h_, src_all, dst_all, n, int(chunk))
                return ln(F.relu(lin(out / deg)))
            h = checkpoint(layer, h, use_reentrant=False) if use_ckpt else layer(h)
        return h
