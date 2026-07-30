"""AIA（属性推論）の参照実装: 個体キーの最近傍検索。

薄い攻撃モジュール。母集団条件付きモデル化（QI→SA の推定）と違い、モデルでなく
記憶の検索（C_i 上の最近傍）を測る。QI=`[age,sex,prefecture]`+無害4臨床
（`aia.qi_continuous`, 既定 FPG 除外）で、(sex,prefecture) ブロック内 per-feature-RMS
最近傍を引く。onset 確信度=k-NN(onset) 平均、time 復元=1-NN の `time`（年）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _grouped_indices(df: pd.DataFrame, keys: list[str]) -> dict[tuple, np.ndarray]:
    """`df.groupby(keys).groups` をキー常に tuple へ正規化して返す（keys 長1でも安全、
    ルールブック §4.2 の流儀）。"""
    out: dict[tuple, np.ndarray] = {}
    for k, idx in df.groupby(keys).groups.items():
        kk = k if isinstance(k, tuple) else (k,)
        out[kk] = np.asarray(idx)
    return out


def aia_predict(
    C_i: pd.DataFrame, query_qi: pd.DataFrame, A_bg: pd.DataFrame, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """C_i 最近傍検索で query_qi 各行の onset 確信度・time 復元値を返す（ルールブック §4.2）。

    連続QI(`aia.qi_continuous`)を A_bg の SD で標準化し、`aia.block_keys`
    （既定 [sex, prefecture]）でブロック化してブロック内 per-feature-RMS 距離の最近傍を引く。
    ブロックが空なら先頭キー（sex）のみのグループへ、それも空なら全体へフォールバックする。
    onset 確信度=k-NN(`aia.k_nn_onset`)の onset 平均、time 復元=1-NN の `time`（年）。
    """
    rcfg = cfg["aia"]
    cont_cols = list(rcfg["qi_continuous"])
    block_keys = list(rcfg["block_keys"])
    k = int(rcfg["k_nn_onset"])

    std = A_bg[cont_cols].std().to_numpy(dtype=float)
    std = np.where(std > 0.0, std, 1.0)

    c_cont = C_i[cont_cols].to_numpy(dtype=float) / std
    q_cont = query_qi[cont_cols].to_numpy(dtype=float) / std
    c_onset = C_i["onset"].to_numpy(dtype=float)
    c_time = C_i["time"].to_numpy(dtype=float)

    blocks = _grouped_indices(C_i, block_keys)
    sex_key = block_keys[0]
    sex_blocks = _grouped_indices(C_i, [sex_key])
    all_idx = np.arange(len(C_i))

    n = len(query_qi)
    onset_conf = np.empty(n, dtype=float)
    time_hat = np.empty(n, dtype=float)
    block_vals = query_qi[block_keys].itertuples(index=False, name=None)
    sex_vals = query_qi[sex_key].to_numpy()

    for i, bv in enumerate(block_vals):
        idx = blocks.get(tuple(bv))
        if idx is None or len(idx) == 0:
            idx = sex_blocks.get((sex_vals[i],))
        if idx is None or len(idx) == 0:
            idx = all_idx
        d = np.sqrt(np.mean((c_cont[idx] - q_cont[i]) ** 2, axis=1))
        kk = min(k, len(idx))
        nn = idx if kk >= len(idx) else idx[np.argpartition(d, kk - 1)[:kk]]
        onset_conf[i] = float(np.mean(c_onset[nn]))
        time_hat[i] = float(c_time[idx[np.argmin(d)]])
    return onset_conf, time_hat
