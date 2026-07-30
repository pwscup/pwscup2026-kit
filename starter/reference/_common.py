"""参照実装スクリプトの共通部品（配布物の読み込み・スキーマ整形）。

このファイルはスクリプト用のヘルパです。採点そのものは `pwscup2026.kit.selfscore`／
CodaBench 側が行います。ルールと様式の正本はルールブックです。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pwscup2026.common import config as config_mod
from pwscup2026.common import submission_schema
from pwscup2026.kit.validate import KIT_CONFIG_PATH

#: 参照AIA が使うパラメータ。ルールブック §4.2 で開示している値をそのまま置いています
#: （配布 config はキット用の部分木なのでこの3つを持っていません）。
AIA_PARAMS = {
    "qi_continuous": ["age", "SBP", "TG", "HDL", "ALT"],
    "block_keys": ["sex", "prefecture"],
    "k_nn_onset": 5,
}

_CLINICAL = ("BMI", "SBP", "TG", "HDL", "ALT", "FPG")
_BINARY = ("sex", "smoking", "onset", "death")
_TIME_FLOOR_YEARS = 1.0 / 365.25


def load_cfg() -> dict:
    """キット同梱の採点・検証パラメータを読み、参照AIA用のキーを足して返す。"""
    cfg = config_mod.load_config(KIT_CONFIG_PATH)
    cfg.setdefault("aia", {}).update(AIA_PARAMS)
    return cfg


def dist_paths(dist: str | Path) -> Path:
    """`participant_data` を展開したディレクトリを検証して返す。"""
    d = Path(dist)
    missing = [n for n in ("B_practice.csv", "A_bg.csv", "targets") if not (d / n).exists()]
    if missing:
        raise SystemExit(
            f"配布物が見つかりません: {d} に {missing} がありません。\n"
            "キット同梱の participant_data/ を指してください（例: --dist ../participant_data）。"
        )
    return d


def read_b(dist: Path) -> pd.DataFrame:
    return pd.read_csv(dist / "B_practice.csv")


def read_a_bg(dist: Path) -> pd.DataFrame:
    return pd.read_csv(dist / "A_bg.csv")


def target_ids(dist: Path) -> list[str]:
    """`targets/mia_columns.json` の対象チームID（練習では 1〜29）。"""
    obj = json.loads((dist / "targets" / "mia_columns.json").read_text(encoding="utf-8"))
    return [str(k) for k in obj["target_columns"]]


def read_target_c(dist: Path, k: str) -> pd.DataFrame:
    return pd.read_csv(dist / "targets" / f"C_{k}.csv")


def read_challenge(dist: Path, k: str) -> pd.DataFrame:
    return pd.read_csv(dist / "targets" / f"aia_challenge_{k}.csv")


def coerce_to_submission(out: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """任意の防御出力を配布スキーマへ整形する（値域 clip・age 整数・二値 int・time の床と地平）。

    行数は変えません（|C| == |B|）。`pwscup2026.kit.validate` を必ず通すための後段です。
    """
    out = out.copy()
    age_lo, age_hi = (float(v) for v in cfg["population"]["age_range"])
    out["age"] = np.floor(np.clip(out["age"].to_numpy(dtype=float), age_lo, age_hi)).astype(int)
    ranges = cfg["schema"]["ranges"]
    for col in _CLINICAL:
        lo, hi = (float(v) for v in ranges[col])
        out[col] = np.clip(out[col].to_numpy(dtype=float), lo, hi)
    for col in _BINARY:
        out[col] = np.clip(np.rint(out[col].to_numpy(dtype=float)), 0, 1).astype(int)
    onset = out["onset"].to_numpy(dtype=int)
    death = out["death"].to_numpy(dtype=int)
    out["death"] = ((onset == 0) & (death == 1)).astype(int)  # (onset, death) は排他・onset 優先
    horizon = float(cfg["onset"]["horizon_years"])
    out["time"] = np.clip(out["time"].to_numpy(dtype=float), _TIME_FLOOR_YEARS, horizon)
    return out[submission_schema.DISTRIBUTED_COLUMNS].reset_index(drop=True)


def write_c(df: pd.DataFrame, out_path: str | Path) -> None:
    """提出用 `C.csv` を書き出す（`record_id` を落とした13列）。"""
    cols = submission_schema.SUBMITTED_C_COLUMNS
    df[cols].to_csv(out_path, index=False)
    print(f"書き出しました: {out_path}  ({len(df)} 行 × {len(cols)} 列)")
