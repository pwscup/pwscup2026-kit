"""参照防御（加工の実装例）: 配布コホート B から匿名化データ C を作る。

    python reference/run_defense.py --dist ../participant_data --out C.csv

既定は `copula_survival`＝共変量をコピュラで再合成し、転帰（time/onset/death）は
B にフィットした**競合リスク Cox** から共変量条件付きにサンプルします。
`--method copula_synth` は転帰を共変量と独立に引く素朴版で、「FPG が発症に効く」といった
関係が C に残らないため U_spec（特定タスクでの結論一致）が立ちにくくなります。
両方を回して自己採点の内訳を見比べると、有用性の4観点が何を見ているか分かります。

**この実装例は方法を縛るものではありません。** 独自の匿名化を使ってかまいません。
"""

from __future__ import annotations

import argparse

import numpy as np

from pwscup2026.defense import synth as synth_mod

from _common import dist_paths, load_cfg, read_b, write_c

METHODS = {
    "copula_survival": synth_mod.copula_survival,
    "copula_synth": synth_mod.copula_synth,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="参照防御で C.csv を作る")
    ap.add_argument("--dist", required=True, help="participant_data を展開したディレクトリ")
    ap.add_argument("--out", default="C.csv", help="出力する C.csv のパス")
    ap.add_argument("--method", default="copula_survival", choices=sorted(METHODS))
    ap.add_argument("--seed", type=int, default=0, help="乱数シード（再現性のため記録してください）")
    args = ap.parse_args()

    dist = dist_paths(args.dist)
    cfg = load_cfg()
    B = read_b(dist)
    rng = np.random.default_rng(args.seed)

    print(f"参照防御 {args.method} を実行します（B: {len(B)} 行 / seed={args.seed}）...")
    C = METHODS[args.method](B, cfg, rng)
    write_c(C, args.out)
    print(
        "次: 自己採点で弱い観点を見てください\n"
        f"  python -m pwscup2026.kit.selfscore {args.out} --dist {args.dist}"
    )


if __name__ == "__main__":
    main()
