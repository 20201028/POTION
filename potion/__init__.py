# -*- coding: utf-8 -*-
"""POTION: prompt-agnostic encoder + prompt encoder + bidirectional decoder.

论文默认超参数见 PAPER_DEFAULTS；各入口脚本的 argparse 默认值与之相同。
"""

__version__ = "1.0.0"

PAPER_DEFAULTS = {
    "d": 128, "encoder_layers": 3, "decoder_layers": 3, "heads": 4,
    "batch": 8, "epochs": 200, "lr": 1e-3, "weight_decay": 1e-4, "patience": 30,
    "rounds": 2, "seeds": (0, 1, 2),
    "hop_buckets": 5, "radius": 2, "cand_cap": 15000,
    "synth_k": 160, "synth_ratio": 0.5, "synth_attr_max": 64,
    "neg_cover": 2.0 / 3.0, "attr_w": 1.0, "pw_max": 50.0, "dice_lambda": 1.0,
    "M_s": 1000,
}
