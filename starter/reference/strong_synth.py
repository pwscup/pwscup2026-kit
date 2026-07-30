"""強い既存合成器で C を作る（任意・追加依存が要ります）。

    # ARF（adversarial random forests・CPU で動く強い汎用合成器）
    uv sync --extra strong-synth
    python reference/strong_synth.py --dist ../participant_data --method arf_synth --out C.csv

    # CTGAN（深層汎用合成器・torch を引くので重いです）
    uv sync --extra ctgan
    python reference/strong_synth.py --dist ../participant_data --method ctgan_synth --out C.csv

事務局が採点系の厳しさを較正するのに使った「参加者が現実に持ち込むであろう強いツール」の
実装です。どちらも配布形式の B だけを入力にし、全列（共変量＋県＋転帰）を平坦に合成します。

**注意**: 平坦な汎用合成器は、希少タイプ（24人しかいません）が mode collapse で消えやすく、
U_rare が落ちがちです。自己採点で4観点の内訳を確認してください。転帰と共変量の関係を
明示的に保つ `copula_survival`（`run_defense.py` の既定）と見比べると差が見えます。

**この実装例は方法を縛るものではありません。** 独自の匿名化を使ってかまいません。
"""

from __future__ import annotations

import argparse

import numpy as np

# arfpy は numpy>=2 で削除された np.in1d を使うため後方互換パッチ（np.isin と1次元で同値）。
if not hasattr(np, "in1d"):
    np.in1d = np.isin  # type: ignore[attr-defined]

import pandas as pd

from pwscup2026.common import schema

from _common import coerce_to_submission, dist_paths, load_cfg, read_b, write_c

CONT = list(schema.CONTINUOUS)      # age, BMI, SBP, TG, HDL, ALT, FPG
COVAR = list(schema.COVARIATES)     # 上記 + sex, smoking
MODEL_COLS = COVAR + ["prefecture", "time", "onset", "death"]


def _fill_na_like(out: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """合成器が NaN を返した列を B の代表値で埋める（行数を変えないため）。"""
    for c in CONT + ["time"]:
        if out[c].isna().any():
            out.loc[out[c].isna(), c] = float(df[c].median())
    for c in ("sex", "smoking", "onset", "death"):
        if out[c].isna().any():
            out.loc[out[c].isna(), c] = int(round(float(df[c].mean())))
    return out


def arf_synth(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """arfpy（ARF）で全列を平坦に合成する。要 arfpy（extra: strong-synth）。"""
    from arfpy import arf

    n = len(df)
    X = df[MODEL_COLS].copy()
    for c in ("sex", "smoking", "prefecture", "onset", "death"):
        X[c] = pd.Categorical(X[c].astype(str))
    for c in CONT + ["time"]:
        X[c] = X[c].astype(float)
    np.random.seed(int(rng.integers(0, 2**31 - 1)))  # arfpy 内部 sklearn の決定性
    m = arf.arf(x=X, num_trees=30, min_node_size=5, verbose=False)
    m.forde()
    syn = m.forge(n=n)

    out = pd.DataFrame({"record_id": df["record_id"].to_numpy()})
    for c in CONT + ["time"]:
        out[c] = pd.to_numeric(syn[c], errors="coerce").to_numpy()
    for c in ("sex", "smoking", "onset", "death"):
        out[c] = pd.to_numeric(syn[c].astype(str), errors="coerce").to_numpy()
    out["prefecture"] = syn["prefecture"].astype(str).to_numpy()
    return coerce_to_submission(_fill_na_like(out, df), cfg)


def ctgan_synth(df: pd.DataFrame, cfg: dict, rng: np.random.Generator, epochs: int = 300) -> pd.DataFrame:
    """CTGAN（SDV）で全列を平坦に合成する。要 sdv+torch（extra: ctgan）。"""
    import torch
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer

    n = len(df)
    X = df[MODEL_COLS].copy()
    for c in ("sex", "smoking", "onset", "death", "prefecture"):
        X[c] = X[c].astype(str)
    md = Metadata.detect_from_dataframe(X)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    model = CTGANSynthesizer(md, epochs=epochs, verbose=False, cuda=False)
    model.fit(X)
    syn = model.sample(num_rows=n)

    out = pd.DataFrame({"record_id": df["record_id"].to_numpy()})
    for c in CONT + ["time"]:
        out[c] = pd.to_numeric(syn[c], errors="coerce").to_numpy()
    for c in ("sex", "smoking", "onset", "death"):
        out[c] = pd.to_numeric(syn[c], errors="coerce").to_numpy()
    out["prefecture"] = syn["prefecture"].astype(str).to_numpy()
    return coerce_to_submission(_fill_na_like(out, df), cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="強い合成器で C.csv を作る（追加依存が要ります）")
    ap.add_argument("--dist", required=True, help="participant_data を展開したディレクトリ")
    ap.add_argument("--out", default="C.csv", help="出力する C.csv のパス")
    ap.add_argument("--method", default="arf_synth", choices=["arf_synth", "ctgan_synth"])
    ap.add_argument("--seed", type=int, default=0, help="乱数シード（再現性のため記録してください）")
    ap.add_argument("--ctgan-epochs", type=int, default=300, help="ctgan_synth の学習エポック数")
    args = ap.parse_args()

    dist = dist_paths(args.dist)
    cfg = load_cfg()
    B = read_b(dist)
    rng = np.random.default_rng(args.seed)

    print(f"{args.method} を実行します（B: {len(B)} 行 / seed={args.seed}）...")
    if args.method == "arf_synth":
        C = arf_synth(B, cfg, rng)
    else:
        C = ctgan_synth(B, cfg, rng, epochs=args.ctgan_epochs)
    write_c(C, args.out)
    print(
        "次: 自己採点で U_rare が落ちていないか確認してください\n"
        f"  python -m pwscup2026.kit.selfscore {args.out} --dist {args.dist}"
    )


if __name__ == "__main__":
    main()
