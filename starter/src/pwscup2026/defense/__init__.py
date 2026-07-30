"""参照防御（実装例）。ルールブック §8。

- `perturb(df, cfg, strength, rng)` … 連続列にガウスノイズを足す単調なノブ。
- `copula_synth(df, cfg, rng)`      … 共変量をコピュラで再合成（転帰は独立に再サンプル）。
- `copula_survival(df, cfg, rng)`   … 同上だが転帰を共変量条件付きにサンプル（`run_defense.py` の既定）。

方法を縛るものではありません。独自の加工を使ってかまいません。
"""
from __future__ import annotations

from .perturb import perturb
from .synth import copula_survival, copula_synth

__all__ = ["perturb", "copula_synth", "copula_survival"]
