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
        if kk >= len(idx):
            nn = idx
        else:
            # ★同値をまとめる（2026-08-01）。`argpartition` で先頭 k 件だけを取ると、
            # k 番目の距離に同値が並んだとき **どれが選ばれるかが C_i の行順で決まる**。
            # ＝提出CSVの並べ方で攻撃の成績が動く。k 番目と等しい距離は全部入れる。
            kth = np.partition(d, kk - 1)[kk - 1]
            nn = idx[d <= kth]
        # ★平均を取る前に**値でソートする**。同値の集合は行順に依らず同じでも、
        # 足す順序が変われば浮動小数の最下位ビットが動く（実測 8.9e-16）。
        # 判定は `|time_hat − 真値| ≤ 0.5年` の閾値比較なので、最下位ビットの差が
        # 境目で結果を裏返しうる（C2ST が n_jobs でコア数依存になっていたのと同じ形）。
        onset_conf[i] = float(np.mean(np.sort(c_onset[nn])))
        # ★同値をまとめる。最近傍が複数いるときに「先頭の1人」を選ばず、全員の time を平均する。
        # QI が連続値で揃っている本番設定では最近傍は常に一意（実測で候補は中央値1・最大1）
        # なので値は変わらない。効くのは、加工が QI を粗く丸めて同じ値の人を大量に作った場合だけ。
        nearest = idx[d <= d.min()]
        time_hat[i] = float(np.mean(np.sort(c_time[nearest])))
    return onset_conf, time_hat
