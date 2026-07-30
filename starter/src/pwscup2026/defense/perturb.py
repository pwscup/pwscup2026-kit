"""摂動族 perturb(B_i, k)。

k=0 で恒等、k↑ でノイズ増（フロンティアを描くための単調ノブ）。連続列は列ごとσでガウス
ノイズを加えてからレンジ/年齢域へクリップ、sex/smokingは確率min(0.5,0.15·k)でフリップ、
prefectureは保存（摂動の対象にしない）。outcomeはtimeだけ乗法ノイズをかけ、
(onset, death)は保存する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import schema

_TIME_FLOOR_YEARS = 1.0 / 365.25  # time の下限＝1日（0 以下にしない）
_SEX_SMOKING_FLIP_SLOPE = 0.15
_SEX_SMOKING_FLIP_CAP = 0.5
_TIME_NOISE_SCALE = 0.1


def perturb(df: pd.DataFrame, cfg: dict, strength: float, rng: np.random.Generator) -> pd.DataFrame:
    """k=strength ≥ 0 の単調摂動を適用する。"""
    k = float(strength)
    if k < 0:
        raise ValueError(f"perturb strength(k) は0以上でなければならない: {k}")

    n = len(df)
    out = df.copy()

    for col in schema.CONTINUOUS:
        std = float(out[col].std(ddof=0))
        if std <= 0:
            continue
        eps = rng.standard_normal(n)
        out[col] = out[col].to_numpy(dtype=float) + k * std * eps

    age_lo, age_hi = (float(v) for v in cfg["population"]["age_range"])
    age = np.clip(out["age"].to_numpy(dtype=float), age_lo, age_hi)
    out["age"] = np.floor(age).astype(int)

    ranges = cfg["schema"]["ranges"]
    for col in ("BMI", "SBP", "TG", "HDL", "ALT", "FPG"):
        lo, hi = (float(v) for v in ranges[col])
        out[col] = np.clip(out[col].to_numpy(dtype=float), lo, hi)

    flip_p = min(_SEX_SMOKING_FLIP_CAP, _SEX_SMOKING_FLIP_SLOPE * k)
    for col in ("sex", "smoking"):
        vals = out[col].to_numpy(dtype=float)
        if flip_p > 0:
            flips = rng.random(n) < flip_p
            vals = np.where(flips, 1.0 - vals, vals)
        out[col] = vals.astype(int)

    horizon = float(cfg["onset"]["horizon_years"])
    eps_t = rng.standard_normal(n)
    time = out["time"].to_numpy(dtype=float) * (1.0 + k * _TIME_NOISE_SCALE * eps_t)
    out["time"] = np.clip(time, _TIME_FLOOR_YEARS, horizon)

    return out
