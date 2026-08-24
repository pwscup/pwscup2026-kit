"""ローカル自己採点（starter kit 同梱・ルールブック §5.4）。

参加者が **CodaBench に出さずに手元で有用性を採点**するための CLI。CodaBench のライブ採点と
同じ純関数 `scoring.utility.score_utility` を、同じ参照値・同じパラメータで呼ぶ薄いラッパで、
新しい採点数理は一切足さない（サーバとローカルで点がズレない唯一の作り方）。

採点できるもの / できないもの:
- **できる**: 有用性 U と4つの観点（U_gen / U_spec / U_rare / U_valid）とその内訳。
- **できない**: 保護 protection・攻撃力（他チームのコホートや真値が要る＝サーバ側でしか計算できない）。

配布物（`--dist`）から読むもの:
- `B_self.csv`  … 配布Bに真値 `is_rare` を足した拡張版B。**これがあると U_rare を完全再現できる**
                   （prof/onset/c2st は真の希少ラベルを基準に採点するため）。無い場合は `B_practice.csv`
                   で代用し、U_rare は mass_star だけの近似になる（その旨を印字する）。
- `utility_ref.json` … 採点が使う参照値（mu/sigma/band/km_signal_D/km_floor）。km_* はブートストラップ
                   由来で手元再計算だと乱数が合わないため、採点と同じ値を移送している。
- `scoring_config.yaml` … 採点パラメータ（公開部分のみ）。

CLI:
    python -m pwscup2026.kit.selfscore <C.csv または 提出zip> --dist <participant_data> [--json]
Docker:
    docker run --rm -v "$PWD":/w hajimeono/pwscup2026-kit:prelim-attack-20260825 score /w/C.csv --dist /w
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..scoring.result import ReportLevel, json_safe
from ..scoring.utility import UtilityReference, score_utility
from . import validate as validate_mod
from .codabench.scoring_io import _canonicalize_c  # サーバ採点と同一の正準化（単一ソース）

#: 観点の日本語ラベル（表示用。内部の指標名は変えない）。
FACET_LABEL = {
    "U_gen": "U_gen  一般的な有用性",
    "U_spec": "U_spec 特定の解析での有用性",
    "U_rare": "U_rare 希少群の再現",
    "U_valid": "U_valid 妥当性",
}


class SelfScoreError(RuntimeError):
    """配布物が足りない・提出が読めない等、採点を始められないときに投げる。"""


# --------------------------------------------------------------------------- #
# 入力ローダ
# --------------------------------------------------------------------------- #
def load_submitted_c(path: str | Path) -> pd.DataFrame:
    """提出物を読む。`C.csv` そのものでも、提出zip（直下フラット）でも受ける。"""
    p = Path(path)
    if not p.exists():
        raise SelfScoreError(f"提出物が見つかりません: {p}")
    if p.suffix.lower() == ".zip":
        if not zipfile.is_zipfile(p):
            raise SelfScoreError(f"zipとして開けません: {p}")
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.rsplit("/", 1)[-1] == "C.csv"]
            if not names:
                raise SelfScoreError(f"zipの中に C.csv がありません（自己採点は加工提出のみ）: {p}")
            return pd.read_csv(io.BytesIO(z.read(names[0])), encoding="utf-8")
    return pd.read_csv(p, encoding="utf-8")


def load_local_reference(dist_dir: str | Path) -> tuple[pd.DataFrame, UtilityReference, dict, dict]:
    """配布物から (B, UtilityReference, cfg, info) を読む。info は表示用のメタ（B_selfの有無など）。

    `b_is_rare` は **B_self.csv の is_rare 列**から供給する（真値の配布口を1つに絞る）。
    utility_ref.json 側の b_is_rare は参加者配布では null。サーバ側 reference_data を `--dist` に
    渡した場合は JSON 側に真値が入っているのでそれを使う（事務局が同じCLIで検算できる）。
    """
    d = Path(dist_dir)
    if not d.is_dir():
        raise SelfScoreError(f"配布物ディレクトリが見つかりません: {d}")

    ref_json = d / "utility_ref.json"
    if not ref_json.exists():
        raise SelfScoreError(
            f"{ref_json.name} がありません。--dist には配布物フォルダ（participant_data）を指定してください。"
        )
    ref_d = json.loads(ref_json.read_text(encoding="utf-8"))

    cfg_path = d / "scoring_config.yaml"
    if not cfg_path.exists():
        raise SelfScoreError(f"{cfg_path.name} がありません（採点パラメータが読めません）。")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    b_self = d / "B_self.csv"
    b_plain = d / "B_practice.csv"
    if b_self.exists():
        b = pd.read_csv(b_self)
        has_b_self = "is_rare" in b.columns
    elif b_plain.exists():
        b = pd.read_csv(b_plain)
        has_b_self = False
    else:
        raise SelfScoreError(f"B_self.csv も B_practice.csv も見つかりません: {d}")

    b_is_rare = None
    if has_b_self:
        b_is_rare = b["is_rare"].to_numpy().astype(bool)
        b = b.drop(columns=["is_rare"])  # 採点に渡すBは配布形式のまま（真値列を混ぜない）
    elif ref_d.get("b_is_rare") is not None:  # サーバ側 reference_data を渡したとき
        b_is_rare = np.asarray(ref_d["b_is_rare"], dtype=bool)

    if b_is_rare is not None and len(b_is_rare) != len(b):
        raise SelfScoreError(f"is_rare({len(b_is_rare)}行) と B({len(b)}行) の行数が違います。")

    ref = UtilityReference(
        mu=np.asarray(ref_d["mu"], dtype=float),
        sigma=np.asarray(ref_d["sigma"], dtype=float),
        band=dict(ref_d["band"]),
        b_is_rare=b_is_rare,
        km_signal_D=ref_d.get("km_signal_D"),
        km_floor=ref_d.get("km_floor"),
    )
    info = {"b_self": bool(b_is_rare is not None), "n_rows_B": int(len(b))}
    return b, ref, cfg, info


# --------------------------------------------------------------------------- #
# 採点
# --------------------------------------------------------------------------- #
def self_score(c_df: pd.DataFrame, b_df: pd.DataFrame, ref: UtilityReference, cfg: dict) -> dict:
    """有用性を採点して {score, facets, detail} を返す（純関数）。

    採点前に提出Cを CodaBench と同じ手順で正準化する（内容ソート＋record_id採番）。これをしないと
    U_rare の C2ST が行順に依存して同じ内容でも点が動く＝サーバと一致しなくなる。
    """
    c_canon = c_df.drop(columns=["record_id"]) if "record_id" in c_df.columns else c_df
    c_canon = _canonicalize_c(c_canon)  # サーバ採点と同一手順
    res = score_utility(c_canon, b_df, ref, cfg, level=ReportLevel.BREAKDOWN)
    return {
        "utility": float(res.score),
        "facets": {k: float(v) for k, v in res.facets.items()},
        "detail": res.detail or {},
    }


def _format_report(out: dict, info: dict, rare_diag: dict | None) -> str:
    lines = [
        "=" * 62,
        "PWS Cup 2026 ローカル自己採点（有用性）",
        "=" * 62,
        f"  有用性 U = {out['utility']:.4f}",
        "",
        "  観点別（Uは4観点の最小値）:",
    ]
    worst = min(out["facets"], key=lambda k: out["facets"][k]) if out["facets"] else None
    for k, v in out["facets"].items():
        mark = "  ← ここが最も弱い" if k == worst else ""
        lines.append(f"    {FACET_LABEL.get(k, k):<26} {v:.4f}{mark}")

    if rare_diag is not None:
        head = f"  希少検出: C={rare_diag['rare_detected']}件 / 閾値{rare_diag['gate']}件"
        lines.append("")
        if rare_diag["below_gate"]:
            lines.append(f"{head} → 希少群の忠実度を検証できず U_rare は 0 になります（ルールブック §6.1.3）。")
        else:
            lines.append(f"{head} → 希少件数はゲートを満たしています。")

    if not info.get("b_self"):
        lines += [
            "",
            "  [注意] B_self.csv（真値 is_rare 付き）が無いため U_rare は近似値です。",
            "         配布物の B_self.csv を --dist のフォルダに置くと CodaBench と同じ点数になります。",
        ]
    lines += [
        "",
        "  ※保護(protection)と攻撃力は他チームの情報が要るため手元では採点できません。",
        "  ※公式記録は CodaBench の採点結果です（ルールブック §5.4）。",
        "=" * 62,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="ローカル自己採点（有用性）— CodaBenchと同じ採点器を手元で回す（ルールブック§5.4）"
    )
    ap.add_argument("submission", help="C.csv または 提出zip のパス")
    ap.add_argument("--dist", required=True, help="配布物フォルダ（participant_data＝B_self/utility_ref/scoring_config がある側）")
    ap.add_argument("--json", action="store_true", help="人間向けの表でなくJSONで出す（スクリプト用）")
    args = ap.parse_args(argv)

    try:
        c_df = load_submitted_c(args.submission)
        b_df, ref, cfg, info = load_local_reference(args.dist)
        out = self_score(c_df, b_df, ref, cfg)
    except SelfScoreError as e:
        print(f"NG: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — 参加者向けCLIなのでトレースバックを出さず理由だけ返す
        print(f"NG: 採点できませんでした（{type(e).__name__}: {e}）", file=sys.stderr)
        print("    提出物の列や行数が配布Bと合っているか、validate で先に確認してください。", file=sys.stderr)
        return 2

    rare_diag = validate_mod.rare_count_diagnostic(c_df, validate_mod.load_dist(args.dist))
    if args.json:
        print(json.dumps(json_safe({**out, "b_self": info["b_self"], "rare_gate": rare_diag}),
                         ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(_format_report(out, info, rare_diag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
