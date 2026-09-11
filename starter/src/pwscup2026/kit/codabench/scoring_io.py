"""採点ロジック（サーバ採点とローカル自己採点の単一ソース）。

提出frames（検証済）＋**真値ディレクトリ（truth_dir）**＋cfg から、リーダーボード用スコアを計算する。
**既存 scoring/ 純関数の薄ラッパ**で、新たな採点数理は足さない（単一ソース）。

★引数名を `ref_dir` から `truth_dir` に
改めた。本番では真値は reference_data ではなく worker ローカルの `/app/data` から読む。
**関数のロジックは変えていない**＝渡されたディレクトリを読むだけ。呼び出し側（`run._run`）が
truth_dir を決める（`/app/data` に `tokens.csv` があればそこ・無ければ従来どおり `$input/ref`。
`PWSCUP_TRUTH_DIR` は明示の上書き）。

- 加工（防御）: `utility`（score_utility）＋`protection`（1−参照攻撃器のMIA被攻撃度）。
- 攻撃: `mia_power`（各ターゲット列の score_mia を集約）＋`aia_power`（各ターゲットの
  score_aia を集約）。

★集約規約: 複数ターゲットに跨る攻撃力は**ターゲット平均**を採る
（攻撃者が全ターゲットに対して平均どれだけ成功したか）。worst-case/pooled ではなく mean。
希少層 worst-case は各ターゲット内で score_mia/score_aia が取る（既存挙動）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...attack import mia as mia_mod
from ...common.submission_schema import read_scoring_csv
from ...scoring import attack_score
from ...scoring.result import ReportLevel
from ...scoring.utility import UtilityReference, score_utility

#: 本番の目印。**このファイルが truth_dir にあれば本番モード**＝提出の
#: `token.txt` から team_id を引いて、その参加者向けの参照ファイルを選ぶ。無ければ練習モード
#: ＝従来どおりの固定ファイル名（`B_practice.csv` 等）を読む。
#: 本番は 1フェーズ＝1タスク＝1つの reference_data に全参加チームぶんが入るので、
#: **提出者を見分けないと全員が同じ1チームの B と比べられてしまう**。
TOKENS_FILE = "tokens.csv"


# --------------------------------------------------------------------------- #
# 提出者の解決（本番モード）
# --------------------------------------------------------------------------- #
def is_production_reference(truth_dir: str | Path) -> bool:
    """真値ディレクトリが本番レイアウト（チーム別参照＋`tokens.csv`）か。"""
    return (Path(truth_dir) / TOKENS_FILE).is_file()


def load_token_map(truth_dir: str | Path) -> dict[str, int]:
    """`tokens.csv`（列 `token,team_id`）を token→team_id で読む。無ければ空辞書。"""
    p = Path(truth_dir) / TOKENS_FILE
    if not p.is_file():
        return {}
    df = pd.read_csv(p, dtype={"token": str})
    return {str(t).strip(): int(k) for t, k in zip(df["token"], df["team_id"])}


def resolve_team_id(truth_dir: str | Path, token: str | None) -> int | None:
    """提出の token から team_id を引く。練習モード・未知トークンは None。

    ★未知トークンで採点を続けない（別チームの B と比べる事故を防ぐ）。本番モードで None なら
    **採点せずに終了**する＝提出回数を消費させない（ルールブック §5.5）。
    提出物の検証でも同じことを見ているので、通常はここに来る前に弾かれる。
    """
    if token is None:
        return None
    return load_token_map(truth_dir).get(str(token).strip())


def _process_ref_paths(truth_dir: Path, team_id: int | None) -> tuple[Path, Path, Path]:
    """加工採点で**提出者ごとに変わる**3ファイル (B, utility_ref, pool_ids) のパス。

    練習は固定名、本番はチーム番号つき。`A_bg.csv` と参照攻撃者（`B_ref.csv` /
    `pool_ids_ref.csv`）は全チーム共通なので、ここでは扱わない。
    """
    if team_id is None:
        return (truth_dir / "B_practice.csv", truth_dir / "utility_ref.json", truth_dir / "pool_ids_practice.csv")
    return (truth_dir / f"B_{team_id}.csv", truth_dir / f"utility_ref_{team_id}.json",
            truth_dir / f"pool_ids_{team_id}.csv")


# --------------------------------------------------------------------------- #
# reference_data ローダ
# --------------------------------------------------------------------------- #
def load_utility_reference(truth_dir: Path, team_id: int | None = None) -> UtilityReference:
    """truth_dir の `utility_ref[_k].json` から `UtilityReference` を復元する。

    mu/sigma/b_is_rare は配列化、band/scalar はそのまま。事務局が B から計算した値を
    移送しただけで、採点時に再計算はしない。

    ★事務局側（truth_dir）の JSON は `b_is_rare` を**入れて**出す（`export.utility_reference_json(...,
    include_b_is_rare=True)`）。参加者配布は null で、参加者は同じ値を `B_self.csv` の
    `is_rare` 列から供給する＝真値の配布口を1つに絞る設計（ルールブック §5.4）。
    """
    _, util_path, _ = _process_ref_paths(Path(truth_dir), team_id)
    d = json.loads(util_path.read_text(encoding="utf-8"))
    b_is_rare = d.get("b_is_rare")
    return UtilityReference(
        mu=np.asarray(d["mu"], dtype=float),
        sigma=np.asarray(d["sigma"], dtype=float),
        band=dict(d["band"]),
        b_is_rare=(np.asarray(b_is_rare, dtype=bool) if b_is_rare is not None else None),
        km_signal_D=d.get("km_signal_D"),
        km_floor=d.get("km_floor"),
    )


def _canonicalize_c(C: pd.DataFrame) -> pd.DataFrame:
    """提出C（record_id無し13列）を採点用に正準化する。

    内容でソート＋record_id採番＝提出の行順に依存しない正準順にする。score_utility の U_rare
    C2ST が `StratifiedKFold(shuffle=True)` で行順依存のため、これをしないと同一内容でも提出順で
    スコアが動く（実測 Δ最大0.015）。正準化すれば同一内容→同一スコア（公平・再現可能）。
    """
    C = C.sort_values(by=list(C.columns), kind="mergesort").reset_index(drop=True)
    C.insert(0, "record_id", range(len(C)))
    return C


def _finite(x) -> float:
    """NaN/inf を0に潰す（scores.json は NaN を書けない・LBが壊れる）。"""
    v = float(x)
    return v if np.isfinite(v) else 0.0


def _nanmean(values: list[float]) -> float:
    """NaN を無視した平均。全部 NaN（＝どの標的でも希少陽性が無い等）なら0。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return 0.0
    return _finite(np.nanmean(arr[np.isfinite(arr)]))


def _pool_set(csv_path: Path) -> set[int]:
    return set(pd.read_csv(csv_path)["pool_id"].astype(int).tolist())


#: `mia_per_target` の帯の境目。**公開する値**（詳細結果の `constants.mia_band_edges` に出す）。
MIA_BAND_EDGES = (0.25, 0.5, 0.75)


def _mia_band(tpr_at_low_fpr: float) -> int:
    """MIA の TPR@low-FPR を 1〜4 の帯に落とす。**上側を含む**（帯2 = (0.25, 0.5]）。

    ★判定に使うのは `tpr_at_low_fpr`（**全体チャネル**の値）だけ。得点の `score`（= max）に
    **変えてはならない**。根拠は事務局の設計文書
    `docs/task_attack_detail_granularity_仕様.md` §2（配布物・採点イメージには含まれない）。
    実際の歯止めは `test_attack_detail_mia_band_uses_all_channel_not_score`
    （テストは採点イメージにも公開キットにも入らない）。

    NaN 分岐は置かない: `attack_score._tpr_at_fpr` は退化入力でも 0.0 を返し、
    それ以外は有限値の線形補間なので `tpr_at_low_fpr` は NaN にならない。
    """
    v = float(tpr_at_low_fpr)
    lo, mid, hi = MIA_BAND_EDGES
    if v <= lo:
        return 1
    if v <= mid:
        return 2
    if v <= hi:
        return 3
    return 4


def _membership_truth(candidate_pool_ids: np.ndarray, target_pool_set: set[int]) -> np.ndarray:
    """候補行の pool_id が ターゲット pool 集合に入るか（overlap 真値）。"""
    return np.isin(candidate_pool_ids, list(target_pool_set)).astype(int)


# --------------------------------------------------------------------------- #
# 加工（防御）採点
# --------------------------------------------------------------------------- #
def score_process(frames: dict[str, pd.DataFrame], truth_dir: Path, cfg: dict,
                  team_id: int | None = None) -> dict:
    """防御提出 `C.csv` を採点する（加工タスク）。

    - `utility` = score_utility(C, 提出者のB, utility_ref, cfg) の集約スカラー。
    - `protection` = 1 − 参照攻撃器（overlap_mia）を C にぶつけた membership 露出（score_mia）。
      候補=`B_ref`、真値=B_ref の pool_id が**提出者の** pool 集合に入るか、希少 worst-case 付き。

    `team_id=None` は練習モード（固定ファイル名）。本番は `token.txt` から解決した team_id が
    渡され、提出者ごとに参照を選ぶ。
    """
    truth_dir = Path(truth_dir)
    C = _canonicalize_c(frames["C.csv"])  # 提出順に依存しない正準順（再現性）

    b_path, _, submitter_pool_path = _process_ref_paths(truth_dir, team_id)
    b_submitter = read_scoring_csv(b_path)
    a_bg = read_scoring_csv(truth_dir / "A_bg.csv")
    util_ref = load_utility_reference(truth_dir, team_id)
    util_ref.a_bg = a_bg  # U_valid の novel（超過質量 D+）が使う。他のファセットは触らない。

    util = score_utility(C, b_submitter, util_ref, cfg, level=ReportLevel.BREAKDOWN)

    # protection: 参照攻撃器 vs 参加者の C（本番匿名性の単体代理・決定(b)）
    # ★参照攻撃者には、どのチームにも配っていない余りのコホートを充てる。
    #   実参加チームを流用すると、そのチームの保護を自分のコホートとの重なりで測ることになる。
    b_ref = read_scoring_csv(truth_dir / "B_ref.csv")
    ref_pool = pd.read_csv(truth_dir / "pool_ids_ref.csv")  # record_id, pool_id, is_rare（B_ref行順）
    submitter_pool_set = _pool_set(submitter_pool_path)

    confidence = mia_mod.overlap_mia(C, a_bg, b_ref, cfg)
    truth = _membership_truth(ref_pool["pool_id"].to_numpy(dtype=int), submitter_pool_set)
    rare_mask = ref_pool["is_rare"].to_numpy(dtype=bool) if "is_rare" in ref_pool else None
    mia_res = attack_score.score_mia(confidence, truth, cfg, rare_mask=rare_mask)
    protection = 1.0 - float(mia_res.score)

    u_rare_d = (util.detail or {}).get("U_rare", {}) if util.detail else {}
    facets = {k: _finite(v) for k, v in util.facets.items()}
    return {
        "utility": float(util.score),
        # ★4ファセットをリーダーボード列に出すためトップレベルへ。
        #   値は score_utility の facets そのもの（詳細結果の内訳と同一値）。
        "U_gen": facets.get("U_gen", 0.0),
        "U_spec": facets.get("U_spec", 0.0),
        "U_rare": facets.get("U_rare", 0.0),
        "U_valid": facets.get("U_valid", 0.0),
        "protection": protection,
        "_breakdown": {
            "utility_facets": {k: float(v) for k, v in util.facets.items()},
            "u_rare_gate": {  # ★機能: 希少件数ゲートの診断（参加者が即座に確認できる構造化フィールド）
                "rare_detected_C": u_rare_d.get("rare_detected_C"),
                "rare_gate": u_rare_d.get("rare_gate"),
                "rare_true_B": u_rare_d.get("rare_true_B"),
                "gated_axes": u_rare_d.get("u_rare_gated"),
            },
            "reference_attack_mia": float(mia_res.score),
            "reference_attack_breakdown": {k: float(v) for k, v in mia_res.breakdown.items()},
        },
    }


# --------------------------------------------------------------------------- #
# 攻撃採点
# --------------------------------------------------------------------------- #
def _attacker_pool(truth_dir: Path, team_id: int | None = None) -> pd.DataFrame:
    """攻撃者コホート P の record_id→pool_id/is_rare（F_mia の行キー照合・rare_mask 用）。

    練習は固定名 `pool_ids_attacker.csv`。本番は提出者自身のコホート＝`pool_ids_{team_id}.csv`
    （対象側と同じファイルを兼用できる＝truth_dir に攻撃者専用のコピーを置かなくてよい）。
    """
    name = "pool_ids_attacker.csv" if team_id is None else f"pool_ids_{team_id}.csv"
    return pd.read_csv(truth_dir / name)  # record_id, pool_id, is_rare


def score_mia_power(f_mia: pd.DataFrame, truth_dir: Path, cfg: dict, team_id: int | None = None) -> dict:
    """MIA密行列を採点する。各ターゲット列 k で score_mia を取り、ターゲット平均を power とする。

    行キー＝攻撃者コホート P の record_id。列 k のセル＝参加者が申告した membership 確信度。
    真値 O_Pk[r] = pool_id(P 行r) ∈ set(pool_id ∈ target k)。希少 worst-case は score_mia が取る。

    本番モード（`team_id` が解決できたとき）は**提出者自身の標的を平均から外す**。
    条文（ルールブック §攻撃の測り方）が母数を「自分のチーム以外の全チーム」と定めているため。
    検証器は自列入りを受理する（提出物を作り直させない）ので、除外はここで行う。
    """
    attacker_pool = _attacker_pool(truth_dir, team_id)
    # F_mia の record_id 順に攻撃者 pool を並べ替え（検証器が行キー一致は保証済）
    ap = attacker_pool.set_index("record_id")
    rid = f_mia["record_id"].to_numpy(dtype=int)
    unknown = sorted(set(rid.tolist()) - set(ap.index.tolist()))
    if unknown:  # 検証器が弾くはずだが、素のKeyErrorで採点ジョブを落とさない
        raise ValueError(f"F_mia: 配布コホートに無い record_id が含まれる: {unknown[:5]}（全{len(unknown)}件）")
    cand_pool_ids = ap.loc[rid, "pool_id"].to_numpy(dtype=int)
    rare_mask = ap.loc[rid, "is_rare"].to_numpy(dtype=bool) if "is_rare" in ap else None

    target_cols = [c for c in f_mia.columns if c != "record_id"]
    if team_id is not None:
        target_cols = [c for c in target_cols if int(c) != int(team_id)]
    # ★2026-08-13: 標的別は**実数をやめて 1〜4 の帯**にする。標的別に細かい数値を返すと、
    #   匿名化データを使わずに採点結果だけから答えを絞り込めてしまうため
    #   （詳しくは docs/task_attack_detail_granularity_仕様.md §1）。
    #   集約に必要な成分は別に持ち回る。
    per_score: list[float] = []
    per_all: list[float] = []
    per_rare: list[float] = []
    per_auc: list[float] = []
    per_band: dict[str, int] = {}
    for col in target_cols:
        k = int(col)
        target_set = _pool_set(truth_dir / f"pool_ids_{k}.csv")
        truth = _membership_truth(cand_pool_ids, target_set)
        conf = pd.to_numeric(f_mia[col], errors="coerce").to_numpy(dtype=float)
        res = attack_score.score_mia(conf, truth, cfg, rare_mask=rare_mask)
        tpr_all = float(res.breakdown.get("tpr_at_low_fpr", float("nan")))
        per_score.append(float(res.score))
        per_all.append(tpr_all)
        per_rare.append(float(res.breakdown.get("tpr_rare", float("nan"))))
        per_auc.append(float(res.breakdown.get("auc", float("nan"))))
        # ★帯は **tpr_all（全体チャネル）** で決める。res.score（= max）ではない。
        #   変更しないこと。根拠は `_mia_band` の docstring が指す設計文書を参照。
        per_band[str(k)] = _mia_band(tpr_all)

    power = float(np.mean(per_score)) if per_score else 0.0
    # ★全体/希少の内訳をLB列に出す。ターゲット平均。
    #   希少側は標的によっては陽性0でNaNになるので nanmean（全NaNなら0）。
    return {"power": power,
            "all": _nanmean(per_all), "rare": _nanmean(per_rare),
            "auc": _nanmean(per_auc), "per_target_band": per_band}


def _aia_floor_by_target(truth_dir: Path) -> dict[int, float]:
    """truth_dir の `aia_floor.csv`（team_id,aia_floor）を読む。無ければ空＝床0扱い。

    この表は `tokens.csv` と同じ側（事務局の真値ディレクトリ）にしか
    無い＝参加者の配布物には入らないので全体表が漏れない（09-07の開示方式そのまま）。
    """
    path = truth_dir / "aia_floor.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(r["team_id"]): float(r["aia_floor"]) for _, r in df.iterrows()}


def score_aia_power(f_aia: pd.DataFrame, truth_dir: Path, cfg: dict, team_id: int | None = None) -> dict:
    """AIAロングを採点する。ターゲット k ごとに aia_truth_k で member/control に割り、
    score_aia を取り、ターゲット平均を power とする。

    本番モード（`team_id` が解決できたとき）は**提出者自身の標的を平均から外す**。
    条文（ルールブック §攻撃の測り方）が母数を「自分のチーム以外の全チーム」と定めているため。
    検証器は自列入りを受理する（提出物を作り直させない）ので、除外はここで行う。

    ★2026-09-08 追加（§2.5）: AIAの帰無の床を**攻撃力側にも**引く。標的kごとに
    `score_aia = clip(score_aia - aia_floor[k], 0, 1)` を適用してから平均する（匿名性側の
    `aggregate_anonymity(aia_floor=...)` と対称）。`truth_dir/aia_floor.csv` が無いときは
    床0＝従来と同じ挙動（後方一致）。
    """
    aia_floor = _aia_floor_by_target(truth_dir)
    # AIA の標的別は従来どおり score の実数を返す（帯にするのは MIA だけ）。
    per_target: dict[str, float] = {}
    per_all: list[float] = []
    per_rare: list[float] = []
    per_p_m: list[float] = []
    per_p_c: list[float] = []
    per_p_m_rare: list[float] = []
    per_p_c_rare: list[float] = []
    for k, grp in f_aia.groupby(f_aia["target_k"].astype(int)):
        if team_id is not None and int(k) == int(team_id):
            continue
        truth = read_scoring_csv(truth_dir / f"aia_truth_{int(k)}.csv")  # challenge_row_id,label,onset,time,is_rare
        merged = grp.merge(truth, on="challenge_row_id", how="inner", suffixes=("", "_truth"))
        is_member = merged["label"].to_numpy() == "member"
        is_control = merged["label"].to_numpy() == "control"

        time_hat = pd.to_numeric(merged["time_hat"], errors="coerce").to_numpy(dtype=float)
        t_onset = merged["onset"].to_numpy(dtype=int)
        t_time = merged["time"].to_numpy(dtype=float)
        is_rare = merged["is_rare"].to_numpy(dtype=bool)

        res = attack_score.score_aia(
            None, time_hat[is_member], t_onset[is_member], t_time[is_member],
            None, time_hat[is_control], t_onset[is_control], t_time[is_control],
            cfg,
            member_rare_mask=is_rare[is_member],
            control_rare_mask=is_rare[is_control],
        )
        floored_score = float(np.clip(res.score - aia_floor.get(int(k), 0.0), 0.0, 1.0))
        per_target[str(int(k))] = floored_score
        per_all.append(float(res.breakdown.get("r_time_all", float("nan"))))
        per_rare.append(float(res.breakdown.get("r_time_rare", float("nan"))))
        per_p_m.append(float(res.breakdown.get("succ_time_member", float("nan"))))
        per_p_c.append(float(res.breakdown.get("succ_time_control", float("nan"))))
        per_p_m_rare.append(float(res.breakdown.get("succ_time_member_rare", float("nan"))))
        per_p_c_rare.append(float(res.breakdown.get("succ_time_control_rare", float("nan"))))

    power = float(np.nanmean(list(per_target.values()))) if per_target else 0.0
    return {"power": power, "per_target": per_target,
            "all": _nanmean(per_all), "rare": _nanmean(per_rare),
            # 教材が「詳細結果に表示されます」と書いている p_m/p_c（標的平均）。
            "p_m": _nanmean(per_p_m), "p_c": _nanmean(per_p_c),
            "p_m_rare": _nanmean(per_p_m_rare), "p_c_rare": _nanmean(per_p_c_rare)}


def score_attack(frames: dict[str, pd.DataFrame], truth_dir: Path, cfg: dict,
                 team_id: int | None = None) -> dict:
    """攻撃提出（`F_mia.csv`＋`F_aia.csv`）を採点する（攻撃タスク）。

    1提出で MIA力・AIA力の2スコア列を返す（攻撃は1提出口）。

    `team_id` は**攻撃者自身**の解決に使う（`F_mia` の行キー＝自分のコホートの record_id）。
    対象側の `pool_ids_{k}.csv` / `aia_truth_{k}.csv` は元から k 付きなので変わらない。
    """
    truth_dir = Path(truth_dir)
    mia = score_mia_power(frames["F_mia.csv"], truth_dir, cfg, team_id)
    aia = score_aia_power(frames["F_aia.csv"], truth_dir, cfg, team_id)
    return {
        "mia_power": mia["power"],
        "aia_power": aia["power"],
        # ★LB列用。合計＝MIA力＋AIA力（∈[0,2]・2025の攻撃得点と同型）。
        "attack_power": _finite(mia["power"]) + _finite(aia["power"]),
        "mia_all": mia["all"], "mia_rare": mia["rare"],
        "aia_all": aia["all"], "aia_rare": aia["rare"],
        # ★2026-08-13: 詳細結果の内訳は成分別の集約＋標的別。MIA の標的別は**実数でなく帯**。
        #   標的別に細かい数値を返すと、匿名化データを使わずに採点結果だけから答えを
        #   絞り込めてしまうため（docs/task_attack_detail_granularity_仕様.md §1）。
        #   帯に粗くすれば、自分の攻撃がどの標的に効いたかは分かるまま、その読み取りには
        #   使えなくなる。得点そのものは一切変えていない（表示だけの変更）。
        "_breakdown": {
            "mia": {
                "tpr_at_low_fpr": mia["all"],   # LB内訳1（MIA全体）と同じ値
                "tpr_rare": mia["rare"],        # LB内訳2（MIA希少）と同じ値
                "auc": mia["auc"],              # 標的ごとの AUC の平均（採点には使わない診断値）
            },
            # 標的ごとの 1〜4 の帯。**全体チャネル tpr_at_low_fpr で判定**（score ではない）。
            "mia_per_target": mia["per_target_band"],
            "aia": {
                "r_time_all": aia["all"],       # LB内訳3（AIA全体）と同じ値
                "r_time_rare": aia["rare"],     # LB内訳4（AIA希少・z·SE デッドゾーン後）と同じ値
                # 成功率そのもの（標的平均）。R は (p_m − p_c) を (1 − p_c) で正規化した値なので、
                # この2つを見れば「当たったが対照も当たっていた」のか「本当に会員だけ当てた」のか判る。
                # ★注意: これは**標的ごとの成功率の平均**で、r_time_* は**標的ごとの R の平均**。
                #   比の平均 ≠ 平均の比 なので (p_m − p_c)/(1 − p_c) は r_time_all に厳密一致しない
                #   （目安として読む値。標的1つぶんの内訳を出すと標的別に戻ってしまうため出さない）。
                "p_m": aia["p_m"], "p_c": aia["p_c"],
                "p_m_rare": aia["p_m_rare"], "p_c_rare": aia["p_c_rare"],
            },
            # AIA の標的別は従来どおり score の実数。
            "aia_per_target": aia["per_target"],
            "constants": {
                "low_fpr": float(cfg["scoring"]["low_fpr"]),
                "tau_time": float(cfg["aia"]["tau_time"]),
                "aia_rare_se_z": float(cfg["scoring"].get("aia_rare_se_z", 0.0)),
                "mia_band_edges": list(MIA_BAND_EDGES),
            },
        },
    }
