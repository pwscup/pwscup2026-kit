"""参照AIA（攻撃の実装例・ルールブック §4.2）: 最近傍で F_aia.csv を作る。

    python reference/run_aia.py --dist ../participant_data --out F_aia.csv

考え方は「相手が実在の個人をほぼそのまま C に残していれば、渡された QI で C の中の
『同じ人』が引ける」です。連続QI（age・SBP・TG・HDL・ALT）を A_bg の SD で標準化し、
(sex, prefecture) のブロック内で最近傍を1件引いて、その行の `time`（追跡年数）を答えます。

出力は **1行が「どの対象チームの・チャレンジの何行目に・何年と答えたか」** のロング形式で、
全対象ぶんを縦に積み上げた1枚です（練習では 326 行 × 29 対象 = 9,454 行）。

**この実装例は方法を縛るものではありません。** 独自の攻撃を使ってかまいません。
"""

from __future__ import annotations

import argparse

import pandas as pd

from pwscup2026.attack import aia as aia_mod

from _common import dist_paths, load_cfg, read_a_bg, read_challenge, read_target_c, target_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="参照AIA（最近傍）で F_aia.csv を作る")
    ap.add_argument("--dist", required=True, help="participant_data を展開したディレクトリ")
    ap.add_argument("--out", default="F_aia.csv", help="出力する F_aia.csv のパス")
    args = ap.parse_args()

    dist = dist_paths(args.dist)
    cfg = load_cfg()
    A_bg = read_a_bg(dist)
    ks = target_ids(dist)

    frames = []
    for k in ks:
        C_k = read_target_c(dist, k)
        chal = read_challenge(dist, k)
        _onset_conf, time_hat = aia_mod.aia_predict(C_k, chal, A_bg, cfg)
        frames.append(
            pd.DataFrame(
                {
                    "target_k": k,
                    "challenge_row_id": chal["challenge_row_id"].to_numpy(),
                    "time_hat": time_hat,
                }
            )
        )
        print(f"  対象 {k}: {len(chal)} 行に回答（time_hat 平均 {time_hat.mean():.2f} 年）")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"書き出しました: {args.out}  ({len(out)} 行)")
    print("F_mia.csv と token.txt と一緒に zip 直下に入れて提出してください。")


if __name__ == "__main__":
    main()
