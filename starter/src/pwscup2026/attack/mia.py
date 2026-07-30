"""MIA（メンバーシップ推論）の参照実装: DOMIAS 密度比。

`score(x) = p_syn(x) / p_ref(x)`（p_syn=C_i上の密度、p_ref=A_bg上の密度）。高比＝C_iが
過学習で記憶したメンバー。連続共変量のみで推定（カテゴリは無視、Bで確定）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from ..common import schema


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _fit_kde_or_none(data: np.ndarray) -> gaussian_kde | None:
    """KDEをfitする。共分散が特異（重い加法ノイズ・過激なマイクロ集約等でC_iが低次元
    部分空間に潰れた場合等）だとscipyがLinAlgErrorを送出するため、Noneで示す
    （判別不能とみなして chance を返す）。
    """
    try:
        return gaussian_kde(data.T)
    except np.linalg.LinAlgError:
        return None


def overlap_mia(C_i: pd.DataFrame, A_bg: pd.DataFrame, B_j: pd.DataFrame, cfg: dict) -> np.ndarray:
    """候補B_j各行のMIA確信度[0,1]を返す。ルールブック §4.1。

    密度比 p_syn/p_ref を KDE（連続共変量, A_bgスケールでz-score標準化）で推定し、
    `ratio/(1+ratio)` で単調に[0,1]へ写像する（順位を保つのでAUC/TPR@FPRには影響しない）。
    どちらかのKDEが特異共分散でfit不能なら判別不能とみなしchance(0.5)を返す。
    """
    estimator = cfg["attack"]["mia_density_estimator"]
    if estimator != "kde":
        raise NotImplementedError(f"attack.mia_density_estimator={estimator!r} は未対応（kdeのみ）")

    cols = schema.CONTINUOUS
    mean = A_bg[cols].mean().to_numpy(dtype=float)
    std = A_bg[cols].std(ddof=0).to_numpy(dtype=float)
    std = np.where(std > 0, std, 1.0)

    syn = _standardize(C_i[cols].to_numpy(dtype=float), mean, std)
    ref = _standardize(A_bg[cols].to_numpy(dtype=float), mean, std)
    target = _standardize(B_j[cols].to_numpy(dtype=float), mean, std)

    kde_syn = _fit_kde_or_none(syn)
    kde_ref = _fit_kde_or_none(ref)
    if kde_syn is None or kde_ref is None:
        return np.full(len(B_j), 0.5)

    p_syn = kde_syn(target.T)
    p_ref = kde_ref(target.T)
    ratio = p_syn / np.clip(p_ref, 1e-300, None)
    return ratio / (1.0 + ratio)
