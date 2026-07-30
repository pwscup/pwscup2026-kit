"""CodaBench 採点アダプタ（薄い I/O 層）。

CodaBench は提出zipを `$input/res/` に、reference_data を `$input/ref/` に展開して
`command $input $output` を呼ぶ（実測のI/O契約）。本モジュールは:

1. `$input/ref/scoring_config.yaml` から採点cfgを読む（reference_dataに同梱＝driftゼロ）。
2. `kit.validate.validate_submission_dir` で2層検証（層1で落ちたら層2は走らせない）。
3. 合格なら `scoring_io.score_process` / `score_attack` を呼ぶ。
4. `$output/scores.json`（リーダーボード用キー）と `detailed_results.html`（内訳・検証メッセージ）を書く。

**リーダーボードは汎用6列 `score_1`〜`score_6`**: CodaBench は
1コンペにLB1つ・scores.json に無いキー列は空欄。加工/攻撃を1コンペに同居させるので、**両方が同じキーを
埋める**ことで空欄を最小化する。写像は

    列      加工（防御）                攻撃
    score_1 有用性 U（Primary）         攻撃力合計（MIA力＋AIA力）
    score_2 U_gen                        MIA攻撃力（全体）
    score_3 U_spec                       MIA攻撃力（希少）
    score_4 U_rare                       AIA攻撃力（全体）
    score_5 U_valid                      AIA攻撃力（希少）
    score_6 保護 protection（参照攻撃1本）  ―（空欄）

`score_6` は**参照攻撃器1本に対する結果であって本番の匿名性ではない**（本番の匿名性は全攻撃者が
出揃ってからのオフライン集約）。ルールブック §5.6。攻撃提出には対応する量が無いので空欄にする。
正確なラベルは詳細結果(detailed_results.html)に出す。

不合格の提出は **非ゼロ終了**（CodaBenchで"Failed"＝提出回数を消費させない）。
scores.json は0で書いておく（リーダーボードの列が壊れないように）が、ジョブ自体は失敗として返す。

エントリ:
    python3 <shim> process  $input $output     # 本番：加工フェーズ
    python3 <shim> attack   $input $output      # 本番：攻撃フェーズ
    python3 <shim> practice $input $output      # 練習フェーズ（C/F を自動判定して振り分け）
shim は scoring_program zip 同梱の3行スクリプト。純関数は焼き込んだパッケージで解決。
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import yaml

from ..validate import validate_submission_dir
from . import scoring_io

#: リーダーボードの汎用6列キー（内訳をリーダーボードで見せる）。
LEADERBOARD_KEYS = ["score_1", "score_2", "score_3", "score_4", "score_5", "score_6"]
#: 提出種別ごとの正式指標名（LEADERBOARD_KEYS と同じ並び）。詳細結果に正式ラベルで出す。
#: None ＝ その列に対応する量が無い（scores.json に出さない＝LBは空欄）。
PROCESS_KEYS = ["utility", "U_gen", "U_spec", "U_rare", "U_valid", "protection"]
ATTACK_KEYS = ["attack_power", "mia_all", "mia_rare", "aia_all", "aia_rare", None]
#: 正式指標名の日本語ラベル（詳細結果表示用）。
_JA_LABEL = {
    "utility": "有用性 U", "protection": "保護 protection（参照攻撃1本・本番の匿名性ではない）",
    "U_gen": "U_gen（汎用的な統計の再現）", "U_spec": "U_spec（結論の一致）",
    "U_rare": "U_rare（希少層の保存）", "U_valid": "U_valid（妥当性）",
    "attack_power": "攻撃力合計（MIA力＋AIA力）",
    "mia_all": "MIA攻撃力（全体）", "mia_rare": "MIA攻撃力（希少）",
    "aia_all": "AIA攻撃力（全体）", "aia_rare": "AIA攻撃力（希少）",
    "mia_power": "MIA攻撃力 mia_power", "aia_power": "属性推論攻撃力 aia_power",
}


def _load_cfg(ref_dir: Path) -> dict:
    return yaml.safe_load((ref_dir / "scoring_config.yaml").read_text(encoding="utf-8"))


def _write_outputs(output_dir: Path, scores: dict[str, float], detail_html: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    # ★日本語を数値文字参照へ。CodaBench は detailed_results.html を charset 宣言
    # 無しで表示するため、ブラウザが既定（windows-1252）で解釈して文字化けする（実測）。
    # 出力を純ASCIIにすればどの charset で読まれても同じに描画される＝宣言に依存しない。
    ascii_html = detail_html.encode("ascii", "xmlcharrefreplace").decode("ascii")
    (output_dir / "detailed_results.html").write_text(ascii_html, encoding="utf-8")


def _detail_ok(title: str, named: dict[str, float], named_keys: list[str | None],
               leaderboard: dict[str, float], breakdown: dict) -> str:
    """検証OKの詳細結果。正式指標名を見出し付きで示し、リーダーボード列との対応も明記。"""
    cells = []
    for lk, nk in zip(LEADERBOARD_KEYS, named_keys):
        if nk:
            cells.append(f"<tr><td>{escape(lk)}</td><td>{escape(_JA_LABEL.get(nk, nk))}</td>"
                         f"<td>{named[nk]:.4f}</td></tr>")
        else:
            cells.append(f"<tr><td>{escape(lk)}</td>"
                         f"<td>（この提出種別では該当なし）</td><td>―</td></tr>")
    rows = "".join(cells)
    bd = escape(json.dumps(breakdown, ensure_ascii=False, indent=2))
    return (
        f"<h2>{escape(title)}</h2><p>検証OK・採点完了。リーダーボードは汎用6列(score_1〜score_6)ですが、"
        f"下表のとおりこのフェーズでの正式な指標に対応します。</p>"
        f"<table border=1 cellpadding=4><tr><th>LB列</th><th>指標</th><th>値</th></tr>{rows}</table>"
        f"<h3>内訳</h3><pre>{bd}</pre>"
    )


def _detail_ng(title: str, errors: list[str]) -> str:
    items = "".join(f"<li>{escape(e)}</li>" for e in errors)
    return (
        f"<h2>{escape(title)}</h2><p style='color:#b00'>提出物が不正です（採点前にreject）。"
        f"以下を直して再提出してください。スコアは全て0です。</p><ul>{items}</ul>"
    )


def _map_to_leaderboard(kind: str, out: dict) -> tuple[dict[str, float], dict[str, float], list[str | None]]:
    """採点純関数の返り値を、リーダーボード(score_1〜score_6)と正式名スコアの両方へ写像する。

    NaN/inf はリーダーボードを壊すので0に潰す（棄権・退化入力の保険）。
    named_keys に None が入っている列は scores.json に出さない＝LBでは空欄になる。
    """
    named_keys = PROCESS_KEYS if kind == "defense" else ATTACK_KEYS
    named = {k: (float(out[k]) if np.isfinite(out[k]) else 0.0) for k in named_keys if k}
    leaderboard = {lk: named[nk] for lk, nk in zip(LEADERBOARD_KEYS, named_keys) if nk}
    return leaderboard, named, named_keys


def _score(kind: str, frames: dict, ref_dir: Path, cfg: dict) -> dict:
    return scoring_io.score_process(frames, ref_dir, cfg) if kind == "defense" \
        else scoring_io.score_attack(frames, ref_dir, cfg)


def _run(mode: str, input_dir: str, output_dir: str) -> int:
    """mode ∈ {process, attack, practice}。practice は提出種別を自動判定して振り分ける。"""
    in_dir = Path(input_dir)
    res_dir = in_dir / "res"
    ref_dir = in_dir / "ref"
    out_dir = Path(output_dir)
    cfg = _load_cfg(ref_dir)

    # 本番は mode 固定（kind_override）、練習は自動判定（kind_override=None）。
    kind_override = {"process": "defense", "attack": "attack"}.get(mode)
    result, pkg = validate_submission_dir(res_dir, cfg, dist_dir=ref_dir, kind_override=kind_override)
    kind = kind_override or pkg.kind
    title = {"process": "加工フェーズ（防御）採点", "attack": "攻撃フェーズ採点"}.get(
        mode, f"練習フェーズ採点（{'防御' if kind == 'defense' else '攻撃' if kind == 'attack' else '種別不明'}）"
    )

    if not result.ok or kind not in ("defense", "attack"):
        # ★検証で落ちた提出は**採点しない＝提出回数を消費させない**。
        # そのためには CodaBench 側で "Failed" にする必要があるので、**非ゼロで終了**する
        # （スコア0で正常終了すると1回ぶん数えられてしまう）。理由は detailed_results と
        # 標準出力の両方に出す。ルールブック §5.5。
        errors = result.errors or [f"提出種が不明: {kind}"]
        _write_outputs(out_dir, {k: 0.0 for k in LEADERBOARD_KEYS}, _detail_ng(title, errors))
        print("NG:", "; ".join(errors))
        return 1

    try:
        out = _score(kind, pkg.frames, ref_dir, cfg)
    except Exception as exc:  # noqa: BLE001
        # 採点器の例外でCodaBenchジョブを落とさない（落ちるとrejectにもならず異常終了する）。
        # スコア0＋理由をdetailedに出す。
        msg = f"採点中に例外が発生しました（提出内容が想定外）: {type(exc).__name__}: {exc}"
        _write_outputs(out_dir, {k: 0.0 for k in LEADERBOARD_KEYS}, _detail_ng(title, [msg]))
        print("ERROR:", msg)
        return 1  # 採点できなかった提出も回数を消費させない（上と同じ理由）
    leaderboard, named, named_keys = _map_to_leaderboard(kind, out)
    _write_outputs(out_dir, leaderboard,
                   _detail_ok(title, named, named_keys, leaderboard, out.get("_breakdown", {})))
    print("OK:", json.dumps(leaderboard))
    return 0


def run_process(input_dir: str, output_dir: str) -> int:
    return _run("process", input_dir, output_dir)


def run_attack(input_dir: str, output_dir: str) -> int:
    return _run("attack", input_dir, output_dir)


def run_practice(input_dir: str, output_dir: str) -> int:
    """練習フェーズ: 提出が C.csv なら加工採点、F_mia/F_aia なら攻撃採点（自動判定）。"""
    return _run("practice", input_dir, output_dir)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] not in ("process", "attack", "practice"):
        print("usage: run.py <process|attack|practice> <input_dir> <output_dir>", file=sys.stderr)
        return 2
    return _run(argv[0], argv[1], argv[2])


if __name__ == "__main__":
    raise SystemExit(main())
