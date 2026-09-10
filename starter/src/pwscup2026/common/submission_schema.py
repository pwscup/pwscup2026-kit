"""配布形式と提出形式の列定義（凍結）。

コホート(B_i)を配布用の単一横持ち表（1人1行・record_id一意）へ射影する `to_distributed` と、
サーバ側の論理整合チェック `validate_submission` を持つ。真値と事務局側の内部列は
配布しません。record_id と真値の対応は事務局だけが保持し、MIA・属性推論の採点に使います。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import schema

_DAYS_PER_YEAR = 365.25


def read_scoring_csv(source, **kwargs) -> pd.DataFrame:
    """採点・検証が読む CSV は必ずこれを通す。

    **これが無いと何が言えなくなるか。** pandas の既定の CSV リーダーは速い近似変換で
    10進表記を double にする。その丸め誤差はビルドに依存するので、**同じファイル・同じ
    pandas でも環境によって別の double になる**ことがある。値の完全一致を数える指標
    （`scoring.utility._no_dup`）では、この差がそのまま得点差になる。

    `float_precision="round_trip"` は書かれた10進表記を**最も近い double へ正しく丸める**ので、
    どの環境でも同じ値になる。書式には依存しない（`41` と `41.0` は同じ double）ので、
    書式を変えて重複判定をすり抜ける手も塞がる（生テキストの比較にはこの性質が無い）。
    """
    kwargs.setdefault("float_precision", "round_trip")
    return pd.read_csv(source, **kwargs)

#: 配布形式（1人1行・record_id一意）の列。
#: QI = age/sex/prefecture。
DISTRIBUTED_COLUMNS = [
    "record_id",
    "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG",
    "time", "onset", "death",
]

#: 提出C_iの列＝配布形式からrecord_idを除いた13列（完全合成で行対応は壊れる＝record_id無意味、
#: ルールブック §5「C_i は record_id を削除」）。
SUBMITTED_C_COLUMNS = [c for c in DISTRIBUTED_COLUMNS if c != "record_id"]

#: (onset, death) の排他規則。両方1は不可。
ALLOWED_ONSET_DEATH = {(0, 0), (1, 0), (0, 1)}


@dataclass
class ValidationResult:
    """`validate_submission` の返り値。純関数・副作用なし。"""

    ok: bool
    errors: list[str] = field(default_factory=list)


def to_distributed(cohort_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[int, int]]:
    """コホート(B_i)を配布形式(distributed_df, truth_map)へ射影する。

    `cohort_df` は事務局が組んだ B_i（`pool_id` と9共変量・prefecture・
    time_to_onset・onset を持つ。`death` は `onset.mortality=lifetable` 生成時のみ存在し、
    無い場合は「死亡なし」＝行政的右打ち切りのみとして扱う。

    **⚠️ `onset/model.py` は既に観測済みsurvival tripleを出力する**（潜在発症時刻 `t_true` は
    打ち切り者では捨てられ、`time_to_onset` は観測exit時刻＝発症時はt_true・打切り時はcensorに
    なる）。よって `time_to_onset` を潜在時刻とみなして `≤horizon` 等で再分類してはいけない
    （打切り者の `time_to_onset=round(horizon)` は常に `≤horizon` になり全員onset=1に誤分類する
    バグを招く）。既存の観測列 `pool.onset`/`pool.time_to_onset`/`pool.death` をそのまま使う。

    record_id はコホート内で新規採番する連番（pool_id とは別・匿名）。
    truth_map: record_id -> pool_id は評価者(𝒮)だけが保持する（MIA/属性推論の真値突合用）。
    """
    cohort_df = cohort_df.reset_index(drop=True)
    n = len(cohort_df)
    record_id = np.arange(n)

    onset = cohort_df["onset"].to_numpy(dtype=int)
    if "death" in cohort_df.columns:
        raw_death = cohort_df["death"].to_numpy(dtype=int)
    else:
        raw_death = np.zeros(n, dtype=int)
    death = ((onset == 0) & (raw_death == 1)).astype(int)  # onset優先で排他を担保（観測表現）

    time_days = cohort_df["time_to_onset"].to_numpy(dtype=float)

    distributed_df = cohort_df[schema.COVARIATES + ["prefecture"]].copy()
    # 凍結スキーマの型(age/sex/smoking=int)に合わせる。内部プールのageは age_bin=[lo,hi+1)
    # の層内一様連続値（covariates/sampler.py, stratum選択の副産物）なので、四捨五入ではなく
    # 満年齢に切り捨てる（[lo,hi+1)→[lo,hi]に収まり、population.age_rangeを超えない）。
    distributed_df["age"] = np.floor(distributed_df["age"].to_numpy(dtype=float)).astype(int)
    distributed_df["sex"] = distributed_df["sex"].to_numpy(dtype=float).astype(int)
    distributed_df["smoking"] = distributed_df["smoking"].to_numpy(dtype=float).astype(int)
    distributed_df.insert(0, "record_id", record_id)

    time_years = np.clip(time_days / _DAYS_PER_YEAR, 1.0 / _DAYS_PER_YEAR, None)
    distributed_df["time"] = time_years
    distributed_df["onset"] = onset
    distributed_df["death"] = death
    distributed_df = distributed_df[DISTRIBUTED_COLUMNS].reset_index(drop=True)

    truth_map = dict(zip(record_id.tolist(), cohort_df["pool_id"].to_numpy().tolist()))
    return distributed_df, truth_map


#: 配布 B_i の行数が分からないとき（`--dist` を付けない自己チェック）だけに使う**緩い**行数上限。
#: 多数派骨格 n=|P_i|+(N_max−1)·c に、希少供給の全体を安全マージンとして足した値。
#: 実際の B_i は 1,049 行なので4倍以上の余裕がある＝**過剰棄却しないための安全弁**であって、
#: 「このくらいの行数まで出してよい」という意味の数字ではない。
#: 本番の提出は配布 B_i の行数と**厳密一致**が要求される（ルールブック §5.1）。
MAX_ROWS_SELFCHECK = 300 + (30 - 1) * 25 + 3500


def _max_rows(cfg: dict) -> int:
    """コホート行数の上限。config に骨格値があればそれで計算し、無ければ上の定数を使う。"""
    acfg = cfg.get("assembler")
    if not acfg or "private_core_size" not in acfg:
        return int(MAX_ROWS_SELFCHECK)
    base = acfg["private_core_size"] + (acfg["n_max"] - 1) * acfg["c"]
    rare_margin = int(cfg.get("run", {}).get("n_rare", 0))
    return int(base + rare_margin)


def _check_row_count(df: pd.DataFrame, cfg: dict, n_expected: int | None, errors: list[str]) -> None:
    """行数ゲート。

    `n_expected`（=配布B_iの行数・評価者/参加者が保持）が与えられたら **|C|==|B_i| を強制**（完全な
    点値テーブルのみ受理・行数削減で"安いprivacy"を得る希釈exploitを封じる）。B_iの正確な行数は
    参加者cfgから出ない（希少配分でチーム毎に変わる）ため引数で渡す。`n_expected=None`（参照が無い
    自己チェック等）のときは従来どおり `_max_rows` の上限のみ（後方互換）。
    """
    if n_expected is not None:
        if len(df) != int(n_expected):
            errors.append(
                f"行数不一致: |C_i|={len(df)} ≠ |B_i|={int(n_expected)}"
                "（提出は配布B_iと同一行数の完全点値テーブルであること）"
            )
    else:
        max_rows = _max_rows(cfg)
        if len(df) > max_rows:
            errors.append(f"行数上限超過: |C_i|={len(df)} > n={max_rows}")


def _check_columns(df: pd.DataFrame, required: list[str], label: str, errors: list[str]) -> bool:
    actual = set(df.columns)
    wanted = set(required)
    if actual == wanted:
        return True
    missing = wanted - actual
    extra = actual - wanted
    if missing:
        errors.append(f"{label}: 列が不足: {sorted(missing)}")
    if extra:
        errors.append(f"{label}: 未知の列: {sorted(extra)}")
    return False


def _is_integral(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isclose(x, np.round(x))


def _check_range(series: pd.Series, lo: float, hi: float, label: str, errors: list[str]) -> None:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if (~np.isfinite(x)).any():
        errors.append(f"{label}: 数値化できない/欠測/非有限(NaN/±inf)がある")
    bad = np.isfinite(x) & ((x < lo) | (x > hi))
    if bad.any():
        errors.append(f"{label}: 値域外[{lo},{hi}]が{int(bad.sum())}行")


def _check_binary(series: pd.Series, label: str, errors: list[str]) -> None:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if (~np.isfinite(x)).any():
        errors.append(f"{label}: 数値化できない/欠測/非有限(NaN/±inf)がある")
    bad = np.isfinite(x) & ~np.isin(x, [0.0, 1.0])
    if bad.any():
        errors.append(f"{label}: {{0,1}}以外の値が{int(bad.sum())}行")


def _check_values(df: pd.DataFrame, cfg: dict, errors: list[str], *, require_record_id: bool = True) -> None:
    """配布形式1表の値レベルチェック（旧`_check_static_values`+`_check_outcome_values`を統合）。

    record_id集合の1:1チェックは、単一表になったことで構造的に発生不能なため
    廃止。`require_record_id=False` で提出 C_i（record_id を削除した13列）にも使える。
    """
    if require_record_id:
        record_id = pd.to_numeric(df["record_id"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(record_id).any() or not _is_integral(record_id).all():
            errors.append("record_id: 整数値ではない/欠測がある")
        if df["record_id"].duplicated().any():
            errors.append("record_id: コホート内一意違反（重複がある）")

    age_lo, age_hi = cfg["population"]["age_range"]
    age = pd.to_numeric(df["age"], errors="coerce").to_numpy(dtype=float)
    if not _is_integral(age).all():
        errors.append("age: 整数値ではない行がある")
    _check_range(df["age"], float(age_lo), float(age_hi), "age", errors)

    _check_binary(df["sex"], "sex", errors)
    _check_binary(df["smoking"], "smoking", errors)

    if df["prefecture"].isna().any() or not df["prefecture"].map(lambda v: isinstance(v, str)).all():
        errors.append("prefecture: 非文字列/欠測がある")

    ranges = cfg["schema"]["ranges"]
    for col in ("BMI", "SBP", "TG", "HDL", "ALT", "FPG"):
        lo, hi = ranges[col]
        _check_range(df[col], float(lo), float(hi), col, errors)

    horizon = float(cfg["onset"]["horizon_years"])
    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    if (~np.isfinite(time)).any():
        errors.append("time: 数値化できない/欠測/非有限(NaN/±inf)がある")
    bad_time = np.isfinite(time) & ((time <= 0.0) | (time > horizon))
    if bad_time.any():
        errors.append(f"time: 0<time≤{horizon}違反が{int(bad_time.sum())}行")

    _check_binary(df["onset"], "onset", errors)
    _check_binary(df["death"], "death", errors)

    onset = pd.to_numeric(df["onset"], errors="coerce")
    death = pd.to_numeric(df["death"], errors="coerce")
    pairs = set(zip(onset.dropna().astype(int), death.dropna().astype(int)))
    bad_pairs = pairs - ALLOWED_ONSET_DEATH
    if bad_pairs:
        n_bad = int(((onset == 1) & (death == 1)).sum())
        errors.append(f"(onset,death): 排他規則違反（onset=death=1）が{n_bad}行")


def validate_submission(df: pd.DataFrame, cfg: dict, *, n_expected: int | None = None) -> ValidationResult:
    """配布提出1表の論理整合をハード棄却する。純関数・副作用なし。

    採点ジョブ起動前に呼ぶ。starter kit にも同梱し、参加者が事前自己チェックに使う
    （ライブ棄却を稀にする）。値レベルチェックは列集合が正しいときのみ行う（列が壊れている
    状態で値チェックしてもノイズにしかならない）。`n_expected` を渡すと行数を |B_i| に強制。
    """
    errors: list[str] = []

    cols_ok = _check_columns(df, DISTRIBUTED_COLUMNS, "distributed", errors)

    _check_row_count(df, cfg, n_expected, errors)

    if cols_ok:
        _check_values(df, cfg, errors)

    return ValidationResult(ok=len(errors) == 0, errors=errors)


def validate_c_submission(df: pd.DataFrame, cfg: dict, *, n_expected: int | None = None) -> ValidationResult:
    """提出C_i（record_id削除の13列）の論理整合をハード棄却する。
    純関数・副作用なし。

    `validate_submission`（配布14列）の提出版。完全合成で行対応が壊れるため参加者の提出C_iは
    record_idを持たない（ルールブック§5）。列集合＝`SUBMITTED_C_COLUMNS`、値域・(onset,death)排他は同一。
    行数は `n_expected`（=配布B_iの行数）を渡すと **|C|==|B_i| を強制**（完全点値テーブル）。
    渡さなければ従来の上限のみ（後方互換）。
    """
    errors: list[str] = []
    cols_ok = _check_columns(df, SUBMITTED_C_COLUMNS, "submitted_C", errors)

    _check_row_count(df, cfg, n_expected, errors)

    if cols_ok:
        _check_values(df, cfg, errors, require_record_id=False)

    return ValidationResult(ok=len(errors) == 0, errors=errors)
