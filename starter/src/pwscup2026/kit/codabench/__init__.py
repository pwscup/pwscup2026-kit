"""CodaBench 採点の配線。

- `scoring_io`: 提出frames＋reference_data＋cfg → スコア（純関数寄り。既存 scoring/ 純関数の薄ラッパ）。
- `run`: CodaBench `$input`/`$output` 契約のI/Oアダプタ（層1・層2検証→採点→scores.json）。

同一採点コードを CodaBench ライブ採点と starter kit 自己チェックの両方で使う。
"""
