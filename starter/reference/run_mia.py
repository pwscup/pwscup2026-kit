"""参照MIA（攻撃の実装例・ルールブック §4.1）: 密度比で F_mia.csv を作る。

    python reference/run_mia.py --dist ../participant_data --out F_mia.csv

考え方は「相手の匿名化データ C の上で密度が高く、背景データ A_bg の上で密度が低い人ほど、
C の元データ B に居た可能性が高い」です。連続共変量で密度比 p_syn/p_ref を KDE で推定し、
`ratio/(1+ratio)` で [0,1] に写します（順位を保つので AUC には影響しません）。

出力は **行＝自分のコホートの候補（1,049 行）× 列＝対象チームID** の密行列です。

**この実装例は方法を縛るものではありません。** 独自の攻撃を使ってかまいません。
"""

from __future__ import annotations

import argparse

import pandas as pd

from pwscup2026.attack import mia as mia_mod

from _common import dist_paths, load_cfg, read_a_bg, read_b, read_target_c, target_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="参照MIA（密度比）で F_mia.csv を作る")
    ap.add_argument("--dist", required=True, help="participant_data を展開したディレクトリ")
    ap.add_argument("--out", default="F_mia.csv", help="出力する F_mia.csv のパス")
    args = ap.parse_args()

    dist = dist_paths(args.dist)
    cfg = load_cfg()
    B_j = read_b(dist)          # 答える候補＝自分のコホート全員
    A_bg = read_a_bg(dist)      # 参照分布
    ks = target_ids(dist)

    out = pd.DataFrame({"record_id": B_j["record_id"].to_numpy()})
    for k in ks:
        C_k = read_target_c(dist, k)
        out[k] = mia_mod.overlap_mia(C_k, A_bg, B_j, cfg)
        print(f"  対象 {k}: 平均確信度 {out[k].mean():.4f}")

    out.to_csv(args.out, index=False)
    print(f"書き出しました: {args.out}  ({len(out)} 行 × {len(out.columns)} 列)")
    print("F_aia.csv と token.txt と一緒に zip 直下に入れて提出してください。")


if __name__ == "__main__":
    main()
