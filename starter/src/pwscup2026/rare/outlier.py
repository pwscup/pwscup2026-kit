"""外れ度 = マハラノビス距離。SETSUBUN (Miura+ 2024) と整合。

M(x) = (x-mu)^T Sigma^-1 (x-mu)。既定 mu,Sigma は多数派ベース（差し替え可）。
SETSUBUN は全データベースで最大Mを worst-case MIAターゲットに選ぶ＝本競技の希少群は
構成上そのターゲットにあたる。連続共変量上で計算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_reference(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """mu, Sigma を推定する。source='majority'(既定)|'full' は呼び出し側が渡すdfで切替。"""
    x = df[columns].to_numpy(dtype=float)
    mu = x.mean(axis=0)
    sigma = np.cov(x, rowvar=False)
    return mu, sigma


def mahalanobis(df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, columns: list[str]) -> np.ndarray:
    """各行のマハラノビス距離。"""
    x = df[columns].to_numpy(dtype=float)
    diff = x - mu
    inv_sigma = np.linalg.pinv(sigma)
    m2 = np.einsum("ij,jk,ik->i", diff, inv_sigma, diff)
    return np.sqrt(np.clip(m2, 0.0, None))
