"""有用性の採点（4ファセット）。

`score_utility(C, B, ref, cfg, level)`:
- `C`, `B` は配布形式の単一df（`to_distributed` 済・`submission_schema.DISTRIBUTED_COLUMNS`・
  真値列なし）。分布系ファセット（U_gen/U_spec/U_valid）はCと同一特徴空間で比較する。
  発症曲線の忠実度（旧 U_time）は U_gen のサブ指標 time_dz として統合（単独facet廃止）。
- `ref` は評価者(𝒮)だけが持つ参照・真値オブジェクト（`UtilityReference`）: Mahalanobis参照
  (mu, Sigma)・外れ値band・（任意）Bの`is_rare`真値マスク（U_rare検出器の妥当性検証用のみ、
  headline採点には使わない）。
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..common import schema
from ..rare import outlier as outlier_mod
from .result import ReportLevel, UtilityResult

_COX_COLUMNS = ["age", "sex", "BMI", "SBP", "log_TG", "HDL", "log_ALT", "FPG", "smoking"]
_CATEGORICAL = ["sex", "smoking", "prefecture"]
_PMSE_FEATURES = list(schema.CONTINUOUS) + ["sex", "smoking"]
_CORR_FEATURES = ["age", "sex", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "smoking"]
# novel（U_valid の4本目）で使う12列。prefecture は除く＝私的コアのせいで A_bg と B の県分布が
# 違い、床が作れないため（2026-08-25 の C2ST と同じ理由）。
_NOVEL_FEATURES = list(schema.CONTINUOUS) + ["time", "sex", "smoking", "onset", "death"]


@dataclass
class UtilityReference:
    """評価者(𝒮)だけが持つ参照。

    mu, sigma: Mahalanobis参照（多数派/A_bgでfit、`rare.outlier.fit_reference`）。
    band: 外れ度バンド（lo/hi/cap）。lo を外れ値検出の閾値に使う。
    b_is_rare: B の `is_rare` 真値マスク（U_rare の採点に使う）
        （U_rareのprof_cal/onset_calがこれをground truthとして使う。echo報酬を避けるため必須）。
        本番採点経路（Sがground truthでrefを構築）でも常に利用可能な前提。
    km_signal_D: D_B=∫(1-S_B)dt（発症曲線サブ指標 time_dz の正規化子・B単独で不変）。
        None なら `_score_u_time` が b から計算する。
    km_floor: 発症曲線L1の標本雑音床（B独立2ブートストラップのL1積分の分位）。この値までは
        KMサンプリング雑音として無罰（デッドゾーン）。Noneなら床=0（デッドゾーン無し）。
    a_bg: 全チーム共通の背景データ（配布形式）。U_valid の `novel` だけが使う。
        参加者も同じファイルを持っているので手元で同じ値を再現できる。None なら novel は無効。
    """

    mu: np.ndarray
    sigma: np.ndarray
    band: dict[str, float]
    b_is_rare: np.ndarray | None = None
    km_signal_D: float | None = None
    km_floor: float | None = None
    a_bg: pd.DataFrame | None = None


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """record_id 順に並べ直すだけ。

    行順を正準化してスコアを再現可能にする（U_rareのC2STが`StratifiedKFold(shuffle=True)`で
    行順依存＝ソートしないと提出順で点が動く）。record_id無しの提出Cは採点アダプタ側で内容ソート＋
    record_id採番して正準化してから渡す（`kit.codabench.scoring_io.score_process`）。
    """
    return df.sort_values("record_id").reset_index(drop=True)


def _tvd(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a.unique()) | set(b.unique()))
    pa = a.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    pb = b.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    return 0.5 * float(np.abs(pa - pb).sum())


def _propensity_scores(x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """傾向スコア p_hat を返す。無penaltyのMLEを第一に、収束失敗/分離（p_hatが非有限や
    mle_retvals.converged=False）のときだけ L2 正則化ロジスティックで必ず収束させて
    「本物の傾向スコア」を返す。

    旧実装は収束失敗時に score=0（最悪ケース）へハード落ちさせていたが、U_gen を min 集約に
    切替えると（U_gen=min）、この偽0が
    facet 全体を 0 に落とし公平性事故になる（実測: k_anon@10 が15コホート中1個だけ
    statsmodels 非収束→pMSE 0.000、他14個は 0.87–0.92）。∴ 収束時の値は不変・失敗時のみ
    正則化 refit で本物の値を返す（他コホートに影響なし・偽0のみ除去）。
    """
    try:
        model = sm.Logit(y, x).fit(disp=0, maxiter=200)
        p_hat = model.predict(x).to_numpy()
        if bool(model.mle_retvals.get("converged", True)) and np.isfinite(p_hat).all():
            return p_hat
    except Exception:
        pass
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, fit_intercept=False)
    clf.fit(x.to_numpy(), y)
    return clf.predict_proba(x.to_numpy())[:, 1]


def _pmse(c_static: pd.DataFrame, b_static: pd.DataFrame) -> tuple[float, float, float]:
    """判別器(ロジスティック回帰)によるpMSE。Woo et al. 2009。outcome除外で直交化。"""
    c_x = c_static[_PMSE_FEATURES].copy()
    b_x = b_static[_PMSE_FEATURES].copy()
    x = pd.concat([c_x, b_x], ignore_index=True)
    y = np.concatenate([np.ones(len(c_x)), np.zeros(len(b_x))])

    for col in schema.CONTINUOUS:
        std = x[col].std(ddof=0)
        x[col] = (x[col] - x[col].mean()) / std if std > 0 else 0.0
    x = sm.add_constant(x, has_constant="add")

    c = float(y.mean())
    ceiling = c * (1.0 - c)
    p_hat = _propensity_scores(x, y)

    pmse = float(np.mean((p_hat - c) ** 2))
    score = float(np.clip(1.0 - pmse / ceiling, 0.0, 1.0)) if ceiling > 0 else 1.0
    return pmse, ceiling, score


def _spearman_corr_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """指定列のSpearman相関行列を返す。"""
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"相関評価に必要な列がありません: {missing}")

    x = df[features].copy()
    for col in features:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    return x.corr(method="spearman").to_numpy(dtype=float)


def _corr_pair_diffs(corr_c: np.ndarray, corr_b: np.ndarray) -> tuple[np.ndarray, dict]:
    """2つの相関行列について、上三角部分の絶対差を返す。

    処理規則:
    - 両方とも有限値: 通常の絶対差を使用する。
    - 両方とも非有限値: 両データで相関が定義できないため、評価対象外とする。
    - 片方だけ非有限値: 相関構造が片側でのみ消失しているため、差を1.0とする。
    """
    if corr_c.shape != corr_b.shape:
        raise ValueError(f"比較する相関行列の形状が一致していません: {corr_c.shape} != {corr_b.shape}")
    if corr_c.ndim != 2 or corr_c.shape[0] != corr_c.shape[1]:
        raise ValueError("corr_cは正方行列である必要があります")
    if corr_b.ndim != 2 or corr_b.shape[0] != corr_b.shape[1]:
        raise ValueError("corr_bは正方行列である必要があります")

    upper = np.triu_indices(corr_c.shape[0], k=1)
    vals_c = corr_c[upper]
    vals_b = corr_b[upper]

    finite_c = np.isfinite(vals_c)
    finite_b = np.isfinite(vals_b)
    both_finite = finite_c & finite_b
    both_invalid = ~finite_c & ~finite_b
    one_invalid = finite_c ^ finite_b

    diffs = np.empty(vals_c.shape, dtype=float)
    diffs[both_finite] = np.abs(vals_c[both_finite] - vals_b[both_finite])
    diffs[one_invalid] = 1.0
    diffs = diffs[~both_invalid]

    detail = {
        "n_pairs_total": int(len(vals_c)),
        "n_pairs_used": int(len(diffs)),
        "n_pairs_both_finite": int(both_finite.sum()),
        "n_pairs_one_invalid": int(one_invalid.sum()),
        "n_pairs_excluded": int(both_invalid.sum()),
    }
    return diffs, detail


def _corr_distance(
    c_static: pd.DataFrame, b_static: pd.DataFrame, features: list[str], tail_quantile: float
) -> tuple[float, float, dict]:
    """C/BのSpearman相関行列差を計算する。

    戻り値:
        mean_dist: 全列ペアの相関差の平均。
        tail_dist: 相関差の上位分位点。一部の大きな相関崩壊が平均で希釈されるのを防ぐ。
        detail: 計算対象ペア数などの詳細。
    """
    if not 0.0 <= tail_quantile <= 1.0:
        raise ValueError(f"tail_quantileは0以上1以下である必要があります: {tail_quantile}")

    corr_c = _spearman_corr_matrix(c_static, features)
    corr_b = _spearman_corr_matrix(b_static, features)
    diffs, pair_detail = _corr_pair_diffs(corr_c, corr_b)

    if len(diffs) == 0:
        # すべてのペアがC/B両方で定義不能の場合、相関に関する情報がないため中立的に距離0とする。
        # 定数化や周辺分布の不一致はmarginal側で評価される。
        mean_dist = 0.0
        tail_dist = 0.0
    else:
        mean_dist = float(np.mean(diffs))
        tail_dist = float(np.quantile(diffs, tail_quantile))

    detail = {
        **pair_detail,
        "mean_dist": mean_dist,
        "tail_dist": tail_dist,
        "tail_quantile": float(tail_quantile),
    }
    return mean_dist, tail_dist, detail


def _score_u_corr(c_static: pd.DataFrame, b_static: pd.DataFrame, cfg: dict) -> tuple[float, dict]:
    """一般集団における列間依存構造（順位相関）の忠実度（案K・2026-08-21導入・2026-09-08目盛り確定）。

    対象は `_CORR_FEATURES`。C/B の Spearman 相関行列の上三角差を平均と上位分位点の
    両方で見て、小さい方を採用する（平均的な再現性と、一部だけ大きく崩れた列ペアの両方を要求）。

    ★`mean_scale`/`tail_scale` は config 上書きに頼らずここの既定値を確定値にする
    （採点イメージと自己採点キットで別々の値になる事故を避ける）。
    """
    corr_cfg = cfg.get("scoring", {}).get("u_gen", {}).get("correlation", {})
    enabled = bool(corr_cfg.get("enabled", True))
    if not enabled:
        return 1.0, {"enabled": False, "score": 1.0}

    features = corr_cfg.get("features") or list(_CORR_FEATURES)
    tail_quantile = float(corr_cfg.get("tail_quantile", 0.90))
    mean_scale = float(corr_cfg.get("mean_scale", 0.40))
    tail_scale = float(corr_cfg.get("tail_scale", 0.80))
    if mean_scale <= 0.0:
        raise ValueError(f"correlation.mean_scaleは正である必要があります: {mean_scale}")
    if tail_scale <= 0.0:
        raise ValueError(f"correlation.tail_scaleは正である必要があります: {tail_scale}")

    observed_mean, observed_tail, dist_detail = _corr_distance(
        c_static, b_static, features=features, tail_quantile=tail_quantile
    )
    mean_score = float(np.clip(1.0 - observed_mean / mean_scale, 0.0, 1.0))
    tail_score = float(np.clip(1.0 - observed_tail / tail_scale, 0.0, 1.0))
    corr_score = float(min(mean_score, tail_score))  # 平均的な相関再現性と局所崩壊の両方を要求

    detail = {
        "enabled": True,
        "score": corr_score,
        "features": list(features),
        "method": "spearman",
        "observed_mean_dist": float(observed_mean),
        "observed_tail_dist": float(observed_tail),
        "mean_score": mean_score,
        "tail_score": tail_score,
        "mean_scale": mean_scale,
        "tail_scale": tail_scale,
        "tail_quantile": tail_quantile,
        **dist_detail,
    }
    return corr_score, detail


# --------------------------------------------------------------------------- #
# U_tail: 裾（外れ度の分布）の忠実度。候補2（2026-08-25 設計・2026-09-02 移植）。
#   採点の marginal は列ごとの KS で、裾は多変量の性質なのでほぼ見えない。U_rare が見ているのは
#   外れ「率」(mass_star) だけで、gate=band.lo を超えた先の形も手前の裾の厚みも見ていない。
#   「周辺分布と相関は保ったまま同時分布の外れ値だけ消す」加工に価格を付けるための軸。
#   床は A_bg の互いに素な標本の対から作った固定表（configs で配る）。A_bg は全チーム同一なので
#   採点側に乱数が入らず、参加者も手元で同じ値を再現できる。
#   ★コード側の既定は `enabled=False`（config を渡さない旧来の呼び出しを壊さないため）。
#   配られる `kit_config.yaml` では `enabled: true` を明示し、U_gen の min-5 に常時入る。
# --------------------------------------------------------------------------- #

_TAIL_LAB = ["BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
_TAIL_UP_Q = (95.0, 99.0, 99.9)
_TAIL_DN_Q = (5.0, 1.0, 0.1)


def _q(q: float) -> str:
    return str(int(q)) if float(q).is_integer() else str(q)


def _tail_stat_names() -> tuple[list[str], list[str]]:
    up = [f"{c}_q{_q(q)}" for c in _TAIL_LAB for q in _TAIL_UP_Q] + [f"maha_q{_q(q)}" for q in _TAIL_UP_Q]
    dn = [f"{c}_q{_q(q)}" for c in _TAIL_LAB for q in _TAIL_DN_Q]
    return up, dn


def _tail_vector(df: pd.DataFrame, ref: "UtilityReference") -> dict[str, float]:
    """裾の統計量。検査値6列の6分位点＋マハラノビス距離の3分位点。参照は ref.mu/ref.sigma のみ。"""
    out: dict[str, float] = {}
    for col in _TAIL_LAB:
        v = df[col].to_numpy(dtype=float)
        for q in (0.1, 1.0, 5.0, 95.0, 99.0, 99.9):
            out[f"{col}_q{_q(q)}"] = float(np.percentile(v, q))
    d = outlier_mod.mahalanobis(df, ref.mu, ref.sigma, schema.CONTINUOUS)
    for q in _TAIL_UP_Q:
        out[f"maha_q{_q(q)}"] = float(np.percentile(d, q))
    return out


def _score_u_tail(
    c_static: pd.DataFrame, b_static: pd.DataFrame, ref: "UtilityReference", cfg: dict
) -> tuple[float | None, dict]:
    """裾の忠実度 U_tail = min(U_shift, U_spike)。

    U_shift（系統的なずれ）＝上側統計量と下側統計量それぞれの平均 z を、帰無での散らばりで
        正規化して合成した shiftz を、Z1 のデッドゾーン付きで K1 の傾きで写す。
    U_spike（局所的な破れ）＝統計量ごとに |ΔS| を max(3σ_床, 0.05×B の IQR) で割り、1 を超えた
        分だけを 10 で頭打ちにして平均し、K2 の傾きで写す。0.05×IQR の歯止めは σ が偶然小さい
        統計量で罰さないため。
    床 σ と較正比は configs の固定表（A_bg 由来・全チーム共通）。採点側に乱数は入らない。
    """
    tail_cfg = (cfg.get("scoring", {}).get("u_gen", {}) or {}).get("tail", {}) or {}
    if not bool(tail_cfg.get("enabled", False)):
        return None, {"enabled": False}
    floor_sd = tail_cfg.get("floor_sd") or {}
    ratio = tail_cfg.get("ratio") or {}
    if not floor_sd:
        raise ValueError("scoring.u_gen.tail.floor_sd が空です（床の固定表が要ります）")
    z1 = float(tail_cfg.get("shift_deadzone_z", 3.0))
    k1 = float(tail_cfg.get("shift_scale", 5.0))
    k2 = float(tail_cfg.get("spike_scale", 0.5))
    cap = float(tail_cfg.get("spike_cap", 10.0))
    iqr_guard = float(tail_cfg.get("iqr_guard", 0.05))
    ratio_max = float(tail_cfg.get("ratio_max", 1.4))
    norm_up = float(tail_cfg.get("shift_norm_up", 0.303))
    norm_dn = float(tail_cfg.get("shift_norm_dn", 0.293))

    up, dn = _tail_stat_names()
    use = [s for s in up + dn if float(ratio.get(s, 9.0)) <= ratio_max]
    t_c = _tail_vector(c_static, ref)
    t_b = _tail_vector(b_static, ref)

    iqr = {col: float(np.percentile(b_static[col], 75) - np.percentile(b_static[col], 25)) for col in _TAIL_LAB}
    md_b = outlier_mod.mahalanobis(b_static, ref.mu, ref.sigma, schema.CONTINUOUS)
    iqr["maha"] = float(np.percentile(md_b, 75) - np.percentile(md_b, 25))

    zs: dict[str, float] = {}
    excess: list[float] = []
    for s in use:
        sig = float(floor_sd[s]) * max(float(ratio.get(s, 1.0)), 1.0)
        delta = t_c[s] - t_b[s]
        zs[s] = delta / sig if sig > 0 else float("nan")
        den = max(3.0 * sig, iqr_guard * iqr[s.split("_q")[0]])
        excess.append(min(max(0.0, abs(delta) / den - 1.0), cap) if den > 0 else 0.0)

    up_used = [s for s in use if s in up]
    dn_used = [s for s in use if s in dn]
    mean_z_up = float(np.nanmean([zs[s] for s in up_used])) if up_used else 0.0
    mean_z_dn = float(np.nanmean([zs[s] for s in dn_used])) if dn_used else 0.0
    shiftz = float(np.sqrt((mean_z_up / norm_up) ** 2 + (mean_z_dn / norm_dn) ** 2) / np.sqrt(2.0))
    spike = float(np.mean(excess)) if excess else 0.0

    u_shift = float(np.clip(1.0 - max(0.0, shiftz - z1) / k1, 0.0, 1.0))
    u_spike = float(np.clip(1.0 - spike / k2, 0.0, 1.0))
    score = float(min(u_shift, u_spike))
    detail = {
        "enabled": True,
        "u_shift": u_shift,
        "u_spike": u_spike,
        "shiftz": shiftz,
        "spike": spike,
        "mean_z_up": mean_z_up,
        "mean_z_dn": mean_z_dn,
        "n_stat": len(use),
    }
    return score, detail


def build_tail_floor(
    a_bg: pd.DataFrame, ref: "UtilityReference", n: int, *, reps: int = 300, seed: int = 20260825
) -> dict[str, float]:
    """裾の床（統計量ごとの sd）を A_bg から作る。configs へ焼く固定表の生成器。

    A_bg の**互いに素な** n 行標本を2本引き、統計量の差を reps 回集めて sd を取る。
    A_bg は全チーム同一ファイルなので、この表はコホートに依らず1本で足りる。
    採点時はこの関数を呼ばず configs の表を読む（採点側に乱数を入れないため）。
    """
    if 2 * n > len(a_bg):
        raise ValueError(f"A_bg の行数 {len(a_bg)} では互いに素な {n} 行標本を2本取れません")
    rng = np.random.default_rng(seed)
    up, dn = _tail_stat_names()
    keys = up + dn
    diffs = np.full((reps, len(keys)), np.nan)
    for r in range(reps):
        idx = rng.permutation(len(a_bg))
        s1 = _tail_vector(a_bg.iloc[idx[:n]].reset_index(drop=True), ref)
        s2 = _tail_vector(a_bg.iloc[idx[n : 2 * n]].reset_index(drop=True), ref)
        diffs[r] = [s1[k] - s2[k] for k in keys]
    return {k: float(v) for k, v in zip(keys, np.nanstd(diffs, axis=0, ddof=1))}


def _score_u_gen(
    c_static: pd.DataFrame,
    b_static: pd.DataFrame,
    time_dz: float,
    time_detail: dict,
    ref: "UtilityReference",
    cfg: dict,
) -> tuple[float, dict]:
    """一般有用性＝min(marginal_mean, corr_score, pmse_score, time_dz, tail_score) の弱点支配集約。

    5指標は役割の異なる相補的な故障検出器（周辺分布 / 順位相関構造 / joint忠実度pMSE / 発症曲線 / 裾）
    ゆえ min で集約。marginal_mean=10列(連続7=1−KS / カテゴリ3=1−TVD)の平均。旧式は marginal_mean と
    marginal_worst を共に平均へ入れ周辺を二重計上していたが、min では marginal_mean のみ採用（marginal は
    worst でなく mean＝k-anon を DP 並みに過剰減点しないため）。marginal_worst はスコアから外し診断 detail
    にのみ残す（ダッシュボード/オフライン集計は BREAKDOWN で各サブ指標を取り出せる）。time_dz は旧 U_time
    （発症フリーKM曲線L1のデッドゾーン正規化）を統合した安全網＝通常1.0、発症構造破壊時のみ沈む。
    """
    marginal: dict[str, float] = {}
    for col in schema.CONTINUOUS:
        stat = float(ks_2samp(c_static[col], b_static[col]).statistic)
        marginal[col] = 1.0 - stat
    for col in _CATEGORICAL:
        marginal[col] = 1.0 - _tvd(c_static[col], b_static[col])

    marginal_mean = float(np.mean(list(marginal.values())))
    marginal_worst = float(np.min(list(marginal.values())))
    pmse, ceiling, pmse_score = _pmse(c_static, b_static)
    corr_score, corr_detail = _score_u_corr(c_static, b_static, cfg)

    parts = [marginal_mean, corr_score, pmse_score, time_dz]
    tail_score, tail_detail = _score_u_tail(c_static, b_static, ref, cfg)
    if tail_score is not None:
        parts.append(tail_score)  # kit_config.yaml の enabled=true で min-5 に入る
    facet = float(min(parts))  # worstはmeanに畳まず診断のみ
    detail = {
        "marginal": marginal,
        "marginal_mean": marginal_mean,
        "marginal_worst": marginal_worst,
        "corr_score": corr_score,
        "correlation": corr_detail,
        "pmse": pmse,
        "pmse_ceiling": ceiling,
        "pmse_score": pmse_score,
        "time_dz": time_dz,
        "time": time_detail,
        "tail_score": tail_score,
        "tail": tail_detail,
    }
    return facet, detail


_LOG_FLOOR = 1e-6  # TG/ALTのschema.ranges下限は0.0で許容されるが、log(0)=-infで
# Cox fit が壊れるため下駄を履かせる（強い摂動をかけた C_i で発見）。


def _cox_frame(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    df["log_TG"] = np.log(np.clip(df["TG"].to_numpy(dtype=float), _LOG_FLOOR, None))
    df["log_ALT"] = np.log(np.clip(df["ALT"].to_numpy(dtype=float), _LOG_FLOOR, None))
    return df[_COX_COLUMNS + ["time", "onset"]]


def _fit_cox(merged: pd.DataFrame) -> CoxPHFitter:
    """素の（正則化なし）部分尤度で Cox をフィットする。収束しない極端なデータ
    （発症者ゼロ・定数列・完全分離など）は例外を送出し、呼び出し側で U_spec=0 にする
    ＝ルールブック §6.1.2「分析が成立しないデータは 0 点」へ実装を合わせる（従来の段階的リッジ
    救済は撤去）。収束不良の ConvergenceWarning も送出へ格上げし、非有限な
    係数/SE や SE=0 の退化も失敗として送出する（＝収束しなかった時を確実にキャッチ）。"""
    cph = CoxPHFitter(penalizer=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        cph.fit(_cox_frame(merged), duration_col="time", event_col="onset")
    params = cph.params_.to_numpy(dtype=float)
    se = cph.standard_errors_.to_numpy(dtype=float)
    if not (np.isfinite(params).all() and np.isfinite(se).all() and bool((se > 0).all())):
        raise ConvergenceError("退化Cox: 非有限な係数/SE、またはSE=0")
    return cph


def _io_overlap(
    lo_b: float, hi_b: float, lo_c: float, hi_c: float, beta_b: float, beta_c: float, se_b: float
) -> float:
    """1係数のIO（信頼区間の重なり）。CI幅退化時は標準化差フォールバック。"""
    width_b, width_c = hi_b - lo_b, hi_c - lo_c
    if width_b > 0 and width_c > 0:
        overlap = min(hi_b, hi_c) - max(lo_b, lo_c)
        io = 0.5 * (overlap / width_b + overlap / width_c)
    else:
        io = 1.0 - min(abs(beta_b - beta_c) / (4.0 * max(se_b, 1e-12)), 1.0)
    return float(np.clip(io, 0.0, 1.0))


def _score_u_spec(c: pd.DataFrame, b: pd.DataFrame) -> tuple[float, dict]:
    """9共変量→onset の多変量 cause-specific Cox を B/C に当て、「分析の結論」を全係数で
    再現できているかを採点（FPG 単独ではなく全9係数の結論一致を見る）。

    各係数について B（本物）と C（合成）の推定を比較し寄与を決める:
      - 有意ステータス（CIが0を除外するか）が食い違う → 0
        （本物で効く効果を消した＝見逃し／本物で効かないのに有意化した＝偽陽性の捏造、の両方を罰する）。
      - 両方有意だが符号が逆 → 0（効果の向きの誤り）。
      - それ以外 → IO（信頼区間の重なり）。両方非有意なら重なるCIでIOは高く出る
        ＝「非有意を非有意と正しく再現した」報酬。
    IOパート = 9係数の寄与の平均。U_spec = min(io_part, tstr_cal)（弱点支配・tstr_cal=÷TRTR自己較正）。
    io_part（係数の忠実度）と tstr（予測の忠実度）は別々の故障モードを捉える直交検出器で、
    binding が防御手法によって入れ替わる＝min が「片方だけ守る」戦略を罰する。

    Cが崩壊した防御（重い加法ノイズ・過激なマイクロ集約・定数列・発症者ゼロ・完全分離等）で
    素のCoxが収束しないデータは「分析が成立しない」とみなし facet=0.0 とする（正則化での救済は
    しない＝ルールブック §6.1.2 に一致）。BもCも素の推定なので参加者の
    手元検算と一致する。
    """
    try:
        cph_c = _fit_cox(c)
        cph_b = _fit_cox(b)
    except (ConvergenceError, ConvergenceWarning, np.linalg.LinAlgError, ValueError, ZeroDivisionError) as exc:
        return 0.0, {"error": f"Cox不成立: {type(exc).__name__}: {exc}", "cox_converged": False}

    per_coef: dict[str, dict] = {}
    contribs: list[float] = []
    for col in _COX_COLUMNS:
        beta_b = float(cph_b.params_[col])
        beta_c = float(cph_c.params_[col])
        se_b = float(cph_b.standard_errors_[col])
        lo_b, hi_b = (float(v) for v in cph_b.confidence_intervals_.loc[col])
        lo_c, hi_c = (float(v) for v in cph_c.confidence_intervals_.loc[col])
        sig_b = (lo_b > 0.0) or (hi_b < 0.0)
        sig_c = (lo_c > 0.0) or (hi_c < 0.0)
        io = _io_overlap(lo_b, hi_b, lo_c, hi_c, beta_b, beta_c, se_b)
        if sig_b != sig_c:
            contrib = 0.0
        elif sig_b and sig_c and (np.sign(beta_b) != np.sign(beta_c)):
            contrib = 0.0
        else:
            contrib = io
        contribs.append(contrib)
        per_coef[col] = {
            "beta_B": beta_b,
            "beta_C": beta_c,
            "sig_B": sig_b,
            "sig_C": sig_c,
            "io": io,
            "contrib": contrib,
        }

    io_part = float(np.mean(contribs))
    # c-index（concordance）は退化したCoxモデル（1行/degenerate C等）でNaNを出し ValueError を
    # 送出しうる。fit失敗と同じ「特異な病的データ＝facet0」として扱い、採点をクラッシュさせない
    # （想定外入力で採点をクラッシュさせないための恒久ガード）。
    try:
        c_index = float(cph_c.score(_cox_frame(b), scoring_method="concordance_index"))
        trtr_c_index = float(cph_b.score(_cox_frame(b), scoring_method="concordance_index"))
    except (ValueError, ConvergenceError, np.linalg.LinAlgError) as exc:
        return 0.0, {"error": f"c-index degenerate: {exc}", "io_part": io_part}
    if not (np.isfinite(c_index) and np.isfinite(trtr_c_index)):
        return 0.0, {"error": "c-index non-finite (degenerate C)", "io_part": io_part}
    tstr_score = float(np.clip(2.0 * (c_index - 0.5), 0.0, 1.0))
    # TRTR自己ベースライン（B学習をBで採点）でtstrをidentity=1へ較正（÷TRTR）。発症モデル自体の
    # concordance天井を除いて io_part と水準を揃え、min集約の退化（tstr天井が全体を支配）を防ぐ。
    # TRTRがchance近傍（予測構造なし）なら較正不能＝tstr_cal=1.0（再現すべき予測信号が無い）。
    denom = trtr_c_index - 0.5
    tstr_cal = 1.0 if denom < 0.02 else float(np.clip((c_index - 0.5) / denom, 0.0, 1.0))

    facet = float(min(io_part, tstr_cal))
    detail = {
        "cox_converged": True,
        "io_part": io_part,
        "per_coefficient": per_coef,
        "tstr_c_index": c_index,
        "tstr": tstr_score,
        "trtr_c_index": trtr_c_index,
        "tstr_cal": tstr_cal,
    }
    return facet, detail


def _km_curve(time: np.ndarray, onset: np.ndarray) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=onset)
    return kmf


def _km_surv_step(time: np.ndarray, onset: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """KM生存関数をgrid点で評価（右連続階段・ベクトル化）。床bootstrap専用の高速版。

    採点本体の integral は lifelines のまま（本関数は雑音閾値スカラーの算出のみに使う、
    U_gen のサブ指標）。
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(onset, dtype=float)
    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    n = len(t)
    if n == 0:
        return np.ones_like(grid, dtype=float)
    uniq, first_idx, counts = np.unique(t, return_index=True, return_counts=True)
    at_risk = n - first_idx
    csum = np.concatenate([[0.0], np.cumsum(e)])
    d = csum[first_idx + counts] - csum[first_idx]
    haz = np.where(at_risk > 0, d / at_risk, 0.0)
    surv = np.cumprod(1.0 - haz)
    idx = np.searchsorted(uniq, grid, side="right") - 1
    return np.where(idx >= 0, surv[np.clip(idx, 0, len(surv) - 1)], 1.0)


def km_signal_strength(time: np.ndarray, onset: np.ndarray, tau: float) -> float:
    """D_B = ∫₀^τ (1 − S_B(t)) dt。Bのみで定まる正規化子（lifelinesで採点本体と整合）。"""
    time = np.asarray(time, dtype=float)
    kmf = _km_curve(time, np.asarray(onset, dtype=float))
    grid = np.union1d(time[time <= tau], [0.0, tau])
    s = kmf.survival_function_at_times(grid).to_numpy()
    return float(np.trapezoid(1.0 - s, grid))


def km_noise_floor(
    time: np.ndarray, onset: np.ndarray, tau: float, nboot: int, pct: float, rng: np.random.Generator
) -> float:
    """B独立2ブートストラップのKM-L1積分の`pct`分位＝標本雑音床（デッドゾーン閾値）。"""
    t = np.asarray(time, dtype=float)
    e = np.asarray(onset, dtype=float)
    n = len(t)
    if n == 0:
        return 0.0
    vals = np.empty(int(nboot))
    for k in range(int(nboot)):
        i1 = rng.integers(0, n, n)
        i2 = rng.integers(0, n, n)
        t1, t2 = t[i1], t[i2]
        grid = np.union1d(np.union1d(t1[t1 <= tau], t2[t2 <= tau]), [0.0, tau])
        s1 = _km_surv_step(t1, e[i1], grid)
        s2 = _km_surv_step(t2, e[i2], grid)
        vals[k] = np.trapezoid(np.abs(s1 - s2), grid)
    return float(np.percentile(vals, pct))


def _score_u_time(c: pd.DataFrame, b: pd.DataFrame, ref: UtilityReference, cfg: dict) -> tuple[float, dict]:
    """発症フリーKM曲線L1一致度を、Bの生存シグナルD_Bで正規化し標本雑音床でデッドゾーン化した
    サブスコア time_dz を返す（U_gen のサブ指標）。
    死亡は競合リスク=打ち切り。integral 自体は現行どおり lifelines KM で計算。"""
    tau = float(cfg["scoring"]["km_tau_years"])
    c_time, b_time = c["time"].to_numpy(), b["time"].to_numpy()
    b_onset = b["onset"].to_numpy()
    kmf_c = _km_curve(c_time, c["onset"].to_numpy())
    kmf_b = _km_curve(b_time, b_onset)

    grid = np.union1d(np.union1d(c_time[c_time <= tau], b_time[b_time <= tau]), [0.0, tau])
    s_c = kmf_c.survival_function_at_times(grid).to_numpy()
    s_b = kmf_b.survival_function_at_times(grid).to_numpy()
    integral = float(np.trapezoid(np.abs(s_b - s_c), grid))

    signal_d = ref.km_signal_D if ref.km_signal_D is not None else km_signal_strength(b_time, b_onset, tau)
    floor = float(ref.km_floor) if ref.km_floor is not None else 0.0
    excess = max(0.0, integral - floor)
    score = float(np.clip(1.0 - excess / max(float(signal_d), 1e-9), 0.0, 1.0))
    return score, {
        "km_l1_integral": integral,
        "km_signal_D": float(signal_d),
        "km_floor": floor,
        "time_dz": score,
        "tau": tau,
    }


def _rare_mask(static_df: pd.DataFrame, ref: UtilityReference) -> np.ndarray:
    """`ref`のμ,Σで配布共変量からMahalanobisを再計算し、band.lo以上を外れ値とする（対称検出）。"""
    m = outlier_mod.mahalanobis(static_df, ref.mu, ref.sigma, schema.CONTINUOUS)
    return m >= float(ref.band["lo"])


def _mass_star(mass_c: float, mass_b: float, n_b: int, z: float) -> tuple[float, float]:
    """外れ率一致（SEデッドゾーン付き）。`(mass_star, se)`を返す。"""
    se = float(np.sqrt(mass_b * (1.0 - mass_b) / n_b)) if n_b > 0 else 0.0
    excess = max(0.0, abs(mass_c - mass_b) - z * se)
    return float(np.clip(1.0 - excess / max(mass_b, 1e-6), 0.0, 1.0)), se


def _prof_raw(x_rare: pd.DataFrame, y_rare: pd.DataFrame, markers: list[str], min_rare_n: int) -> float | None:
    """希少群の臨床プロファイル分布一致（マーカー平均の 1-KS）。"""
    if len(x_rare) < min_rare_n or len(y_rare) < min_rare_n:
        return None
    return float(np.mean([1.0 - ks_2samp(x_rare[m], y_rare[m]).statistic for m in markers]))


def _onset_raw(x_rare: pd.DataFrame, y_rare: pd.DataFrame, min_rare_n: int) -> float | None:
    """希少群の発症率一致。"""
    if len(x_rare) < min_rare_n or len(y_rare) < min_rare_n:
        return None
    return float(np.clip(1.0 - abs(x_rare["onset"].mean() - y_rare["onset"].mean()), 0.0, 1.0))


def _canon_rows(df: pd.DataFrame, markers: list[str]) -> np.ndarray:
    """マーカー値だけで決まる正準行順に並べ替えた行列を返す（＝入力の行順に不変）。

    `np.lexsort` は最後のキーが主キーなので、markers[0] が主キーになるよう逆順に渡す。
    """
    m = df[markers].to_numpy(float)
    return m[np.lexsort(m.T[::-1])]


def _c2st_raw(
    x_rare: pd.DataFrame, y_rare: pd.DataFrame, markers: list[str], min_rare_n: int, n_estimators: int
) -> float | None:
    """希少群の**joint忠実度**＝2標本分類テスト（C2ST）。マーカー空間で x_rare(=C検出希少) と
    y_rare(=B真希少) をRandomForestで判別し、`1 - 2|AUC-0.5|`（判別不能=1=joint一致）を返す。
    prof(1-KS=周辺のみ)が見落とす**同時分布/裾依存**を捉える（copulaのガウス依存が苦手な軸）。
    （希少層の同時分布まで見る軸）。

    決定性: RF/CV とも random_state=0、**RF は n_jobs=1**（下のコメント）、そして
    **入力の行順に不変**（`_canon_rows` で正準ソートしてから積む）。データ僅少（n_splits<2）なら
    None（較正で除外）。

    ★行順不変が要る理由（2026-08-02）。`StratifiedKFold(shuffle=True)` は行の**位置**で fold を
    割るので、同じ行集合でも並びが違えば別の分割になり AUC が動く。U_rare の joint 軸は
    `c2st_cal = c2st_val ÷ c2st_base` で、val は **C 側＝内容ソートで正準化された並び**
    （`kit.codabench.scoring_io._canonicalize_c`）、base は **B 側＝配布時の並び**（恣意的で
    チームごとに違う）。∴ 正準化しないと **C=B の完璧な提出でも比が 1 にならない**。
    本番規模の実測で identity の U_rare が **0.524〜1.000** に振れ（36.7% の並びが 0.95 未満）、
    加工の巧拙と無関係なオフセットが各チームの U の天井を決めていた。C2ST の AUC は2標本の
    **集合としての**性質なので、正準順を1つ選んでも指標の意味は変わらない。
    memory: pwscup2026-c2st-row-order-0802。
    """
    if len(x_rare) < min_rare_n or len(y_rare) < min_rare_n:
        return None
    X = np.vstack([_canon_rows(x_rare, markers), _canon_rows(y_rare, markers)])
    y = np.concatenate([np.ones(len(x_rare)), np.zeros(len(y_rare))])
    n_splits = min(5, int(y.sum()), int((1 - y).sum()))
    if n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    # ★n_jobs=1 必須。-1 だと predict_proba の確率を複数スレッドが共有配列へ
    # 足し込むため、並列数（＝機械のコア数）で加算順が変わり最下位ビットがぶれる。希少は
    # B24人×C36人程度なので AUC の粒度が 1/(36*24)=0.0012 と粗く、同点付近のペアが1組
    # 入れ替わるだけで U_rare が 0.003 動く（実測: Mac 0.3083 / コンテナ 0.3120）。
    # ルールブック§5.4「ローカル採点＝サーバ採点」の前提。木200本・60行なので速度は無関係。
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=1)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    return float(np.clip(1.0 - 2.0 * abs(auc - 0.5), 0.0, 1.0))


def _calibrate(val: float | None, base: float | None) -> float | None:
    """`val`を自己ベースライン`base`(=identityが達成しうる上限)で割ってidentity=1に較正。

    `base`がNoneまたは≈0(較正不能)なら較正済みサブ指標そのものを除外する(生値を混ぜない)。
    """
    if val is None or base is None or base <= 1e-6:
        return None
    return float(np.clip(val / base, 0.0, 1.0))


def _aggregate_u_rare(parts: list[float], method: str) -> float:
    if method == "min":
        return float(min(parts))
    if method == "mean":
        return float(np.mean(parts))
    return float(len(parts) / sum(1.0 / float(np.clip(p, 1e-6, None)) for p in parts))


def _score_u_rare(
    c: pd.DataFrame, b: pd.DataFrame, c_static: pd.DataFrame, b_static: pd.DataFrame, ref: UtilityReference, cfg: dict
) -> tuple[float, dict]:
    """希少(外れ値)の忠実度。mass_star(外れ率一致・SEデッドゾーン)/prof_cal(臨床プロファイル)/
    onset_cal(発症率)の最悪支配的集約（旧 mass+rmst 構成を置換、
    Bの標本実現へのecho報酬を除去）。
    """
    u_cfg = cfg["scoring"]["u_rare"]
    markers = u_cfg["markers"] or list(schema.CONTINUOUS)
    min_rare_n = int(u_cfg["min_rare_n"])
    z = float(u_cfg["mass_se_z"])
    method = u_cfg["aggregate"]
    c2st_cfg = u_cfg.get("c2st") or {}
    # C2STはCV/RFのため周辺系より多くの希少人数を要する。min_rare_nと下限のmaxでゲート。
    c2st_min = max(min_rare_n, int(c2st_cfg.get("min_rare_n", min_rare_n)))
    c2st_trees = int(c2st_cfg.get("n_estimators", 200))

    rare_c = _rare_mask(c_static, ref)
    rare_b = _rare_mask(b_static, ref)

    mass_c, mass_b = float(rare_c.mean()), float(rare_b.mean())
    mass_star, se = _mass_star(mass_c, mass_b, len(rare_b), z)

    prof_val = prof_base = prof_cal = None
    onset_val = onset_base = onset_cal = None
    c2st_val = c2st_base = c2st_cal = None
    n_c_rare = int(rare_c.sum())     # C（提出）で希少と検出された件数
    n_b_det = int(rare_b.sum())      # B（本物）で希少と検出された件数（自己ベースライン用）
    truth = np.asarray(ref.b_is_rare, dtype=bool) if ref.b_is_rare is not None else None
    n_b_truth = int(truth.sum()) if truth is not None else 0  # B の真の希少件数（検証可能性の基準）
    gated_axes: list[str] = []

    def _cal_gated(val: float | None, base: float | None, gate: int, axis: str) -> float | None:
        """較正値を返す。ただし ★次のゲートがある:
        参照(B)側に検証可能な希少がある（真値 n_b_truth ≥ gate かつ B検出 n_b_det ≥ gate）のに、
        C側の検出希少が gate 未満なら **0点**（希少群を再現できていない＝抑制で検証を回避させない）。
        参照側が不足（世界に希少が少ない）なら従来どおり除外(None)。それ以外は自己ベースライン較正。"""
        ref_testable = (n_b_truth >= gate) and (n_b_det >= gate)
        if ref_testable and n_c_rare < gate:
            gated_axes.append(axis)
            return 0.0
        return _calibrate(val, base)

    if truth is not None:
        b_truth = b.loc[truth]
        prof_val = _prof_raw(c.loc[rare_c], b_truth, markers, min_rare_n)
        prof_base = _prof_raw(b.loc[rare_b], b_truth, markers, min_rare_n)
        prof_cal = _cal_gated(prof_val, prof_base, min_rare_n, "prof_cal")

        onset_val = _onset_raw(c.loc[rare_c], b_truth, min_rare_n)
        onset_base = _onset_raw(b.loc[rare_b], b_truth, min_rare_n)
        onset_cal = _cal_gated(onset_val, onset_base, min_rare_n, "onset_cal")

        # C2ST joint忠実度（C検出希少 vs B真希少）。B自己ベースライン(B検出希少 vs B真希少)で較正。
        c2st_val = _c2st_raw(c.loc[rare_c], b_truth, markers, c2st_min, c2st_trees)
        c2st_base = _c2st_raw(b.loc[rare_b], b_truth, markers, c2st_min, c2st_trees)
        c2st_cal = _cal_gated(c2st_val, c2st_base, c2st_min, "c2st_cal")

    parts_used = []
    parts = []
    for name, value in (
        ("mass_star", mass_star),
        ("prof_cal", prof_cal),
        ("onset_cal", onset_cal),
        ("c2st_cal", c2st_cal),
    ):
        if value is not None:
            parts_used.append(name)
            parts.append(value)

    facet = _aggregate_u_rare(parts, method)
    detail = {
        "mass_C": mass_c,
        "mass_B": mass_b,
        "mass_se": se,
        "mass_star": mass_star,
        "prof_val": prof_val,
        "prof_base": prof_base,
        "prof_cal": prof_cal,
        "onset_val": onset_val,
        "onset_base": onset_base,
        "onset_cal": onset_cal,
        "c2st_val": c2st_val,
        "c2st_base": c2st_base,
        "c2st_cal": c2st_cal,
        "aggregate": method,
        "parts_used": parts_used,
        "rare_detected_C": n_c_rare,       # ★機能: 参加者向け診断（Cで希少と判定された件数）
        "rare_detected_B": n_b_det,
        "rare_true_B": n_b_truth,
        "rare_gate": c2st_min,             # ★これ未満だと希少忠実度が0になる閾値（c2st）
        "rare_gate_prof_onset": min_rare_n,
        "u_rare_gated": gated_axes,        # C側不足で0にした軸（[]なら未発火）
    }
    if truth is not None:
        tp = int(np.sum(rare_b & truth))
        detail["b_detector_precision"] = tp / max(int(rare_b.sum()), 1)
        detail["b_detector_recall"] = tp / max(int(truth.sum()), 1)
    return facet, detail


_NO_FAB_GROUPS = ["prefecture", "sex", "smoking"]
_NO_FAB_TARGETS = ["BMI", "SBP", "TG", "HDL", "ALT", "FPG", "onset"]


def _weighted_between_sd(values: np.ndarray, groups: np.ndarray) -> float:
    """群平均の重み付き（群サイズ）標準偏差＝群間ばらつき。単群/空は0（ばらつき無し）。

    within分散に依存しない「群→指標の関係の強さ」の生の大きさ。no_fab はこれを B の固定SDで
    割って C/B を比較する（÷C自身の総SDにしない＝k-anon等の分散縮小で比率が偽増する誤検出を
    避ける）。
    """
    n = len(values)
    if n == 0:
        return 0.0
    frame = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(groups)})
    grp = frame.groupby("g", sort=False)["v"]
    means = grp.mean().to_numpy(dtype=float)
    counts = grp.size().to_numpy(dtype=float)
    total = float(counts.sum())
    if total <= 0.0 or len(means) <= 1:
        return 0.0
    w = counts / total
    mbar = float(np.sum(w * means))
    var = float(np.sum(w * (means - mbar) ** 2))
    return float(np.sqrt(max(var, 0.0)))


def _no_fab(c: pd.DataFrame, b: pd.DataFrame, cfg: dict) -> tuple[float, dict]:
    """関係インフレのペナルティ。実データ(B)より強い
    群→指標の関係を捏造したCを弱くどこかで必ず罰す（疫学的事実に反する加工の抑止）。

    groups{prefecture,sex,smoking}×targets{6連続+onset}の各ペアで
    `assoc = 群平均の重み付きSD ÷ BのSD(固定・Bレンジwinsorize後)` を C/B で計算し、
    片側超過 `max(0, assoc_C - assoc_B - tol)` を `/scale` で [0,1] にクリップ、全ペア平均。
    `no_fab = 1 - 平均`。設計要点3つ（各々D4で実測必要と判明）:
      (1) ÷B固定SD（÷C総SDでない）＝k-anon等の分散縮小での偽増を回避。
      (2) Bレンジwinsorize＝暴れ値の偽広がりを除去。ここでの「Bレンジ」は **B に実際に現れた
          min/max** であって `schema.ranges`（固定の生理的レンジ・validate_submissionのハード棄却）
          ではない。両者には広い隙間があり、ハード棄却を通った値がBレンジ外に出るのは普通に起きる
          （最も極端なのはFPG: schema上限400に対しBのmaxは126前後＝ベースラインが非糖尿病のため）。
          ∴ このwinsorizeは棄却と冗長ではなく、外れ値1本で群間SDが数倍に膨れるのを防いでいる。
      (3) 片側（Bより弱める＝ブラーは無罰）＝copula/ノイズ等legitを守る。
    弱くてよい前提（抑止用途）＝U_valid(mean)に置く。tol/scaleは暫定・config化（調整可）。
    """
    nf_cfg = cfg["scoring"].get("u_valid", {}).get("no_fab", {})
    tol = float(nf_cfg.get("tol", 0.08))
    scale = float(nf_cfg.get("scale", 0.5))

    penalties: list[float] = []
    pairs: dict[str, dict] = {}
    for tgt in _NO_FAB_TARGETS:
        lo, hi = float(b[tgt].min()), float(b[tgt].max())
        b_t = np.clip(b[tgt].to_numpy(dtype=float), lo, hi)  # Bは自レンジ内＝no-op
        c_t = np.clip(c[tgt].to_numpy(dtype=float), lo, hi)  # CをBレンジにwinsorize
        sd_b = float(np.std(b_t))  # 固定参照SD（ddof=0）
        if sd_b <= 1e-9:
            continue  # 参照に信号が無い＝捏造を測れない（skip）
        for grp in _NO_FAB_GROUPS:
            assoc_c = _weighted_between_sd(c_t, c[grp].to_numpy()) / sd_b
            assoc_b = _weighted_between_sd(b_t, b[grp].to_numpy()) / sd_b
            excess = max(0.0, assoc_c - assoc_b - tol)
            pen = float(np.clip(excess / scale, 0.0, 1.0)) if scale > 0 else float(excess > 0)
            penalties.append(pen)
            pairs[f"{grp}:{tgt}"] = {"assoc_C": assoc_c, "assoc_B": assoc_b, "penalty": pen}

    mean_pen = float(np.mean(penalties)) if penalties else 0.0
    score = float(np.clip(1.0 - mean_pen, 0.0, 1.0))
    detail = {"tol": tol, "scale": scale, "mean_penalty": mean_pen, "n_pairs": len(penalties), "pairs": pairs}
    return score, detail


def _no_dup(c: pd.DataFrame, cfg: dict) -> tuple[float, dict]:
    """完全重複行のペナルティ（2026-08-20 追加）。同一個体を多重計上したデータは、行数ぶんの
    独立標本があるという前提を壊す（分散の過小推定・実効n の水増し）＝内部妥当性の問題なので
    U_valid に置く。**実データ B の重複行は全チームで 0 行**（連続7列を持つ13列の完全一致は
    現実には起こらない）＝ tol=0 は「本物には無いものを作った」ことへの罰として素直。

    dup_rate = (n - ユニーク行数)/n。no_dup = 1 - clip((dup_rate - tol)/scale, 0, 1)。
    U_valid は mean 集約なので、これが 0 になっても facet は 3/4 までしか落ちない＝抑止の
    強さは「弱くてよい」（D4 の no_fab と同じ位置づけ）。マイクロ集約系（群中心の繰り返しで
    重複が出る）を過剰に殺さないのはこの mean のおかげ。
    """
    ncfg = cfg["scoring"].get("u_valid", {}).get("no_dup", {})
    tol = float(ncfg.get("tol", 0.0))
    scale = float(ncfg.get("scale", 0.2))

    cols = [col for col in c.columns if col != "record_id"]
    n = len(c)
    n_uniq = len(c[cols].drop_duplicates())
    dup_rate = float((n - n_uniq) / n) if n else 0.0
    pen = float(np.clip((dup_rate - tol) / scale, 0.0, 1.0)) if scale > 0 else float(dup_rate > tol)
    return float(1.0 - pen), {
        "dup_rate": dup_rate,
        "n_dup_rows": int(n - n_uniq),
        "tol": tol,
        "scale": scale,
    }


def _nn_min_dist(X: np.ndarray, A: np.ndarray, block: int = 256) -> np.ndarray:
    """X の各行から A の最近傍までのユークリッド距離。行数が大きいのでブロックで回す。"""
    out = np.empty(len(X), dtype=float)
    for i in range(0, len(X), block):
        d2 = ((X[i : i + block, None, :] - A[None, :, :]) ** 2).sum(-1)
        out[i : i + block] = np.sqrt(d2.min(1))
    return out


def _score_u_novel(
    c: pd.DataFrame, b: pd.DataFrame, a_bg: pd.DataFrame | None, cfg: dict
) -> tuple[float | None, dict]:
    """novel（新規性）= C の A_bg への異常接近が「新しい行を運んでいない」証拠になっていないか、を
    超過質量 D+ で罰する。

    ねらいは旧来の距離版と同じ: 分析者は背景データ A_bg を持っている。C がその部分集合を
    渡すだけなら、分析者の手元は増えない。B の行は A_bg と互いに素なので、正直な C は A_bg への
    最近傍距離の分布が B と同じになる。A_bg の行を選び直した C はこの距離が系統的に小さくなる。

    旧実装（distance floor 版）は「B から A_bg への最近傍距離の下位分位点」という**定数**で
    打切っていた。打切りの位置を人が決めない器として、片側二標本 KS 統計量 D+ に
    差し替えている。

    手続き。
      1. `_NOVEL_FEATURES`（prefecture を除く12列）を A_bg の平均・標準偏差で標準化する。
      2. d_c = C の各行から A_bg への最近傍距離、d_b = B の各行から A_bg への最近傍距離
         （どちらも `_nn_min_dist`）。
      3. D+ = sup_t [F_C(t) − F_B(t)]（経験分布関数の片側二標本KS統計量。C 側の距離分布が
         B より手前（小さい側）へはみ出した質量の最大値＝「A_bg に異常接近した」度合い）。
      4. novel = 1 − clip((D+ − d0) / (1 − d0), 0, 1)。d0 はサンプリング雑音のデッドゾーン
         （既定 `scoring.u_valid.novel.deadzone` = 0.0534＝n=m=1049 の片側KS 5%点）。

    無効なとき（A_bg 不在、または特徴列が足りない）は None を返し、`_score_u_valid` の平均に
    入らない。
    """
    ncfg = (cfg.get("scoring", {}) or {}).get("u_valid", {}).get("novel", {}) or {}
    if a_bg is None:
        return None, {"enabled": False, "reason": "A_bg が渡されていない"}

    feats = list(ncfg.get("features") or _NOVEL_FEATURES)
    missing = [f for f in feats if f not in a_bg.columns or f not in c.columns or f not in b.columns]
    if missing:
        return None, {"enabled": False, "reason": f"列が足りない: {missing}"}

    A = a_bg[feats].to_numpy(dtype=float)
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd = np.where(sd > 0.0, sd, 1.0)  # 定数列でゼロ割りしない
    Az = (A - mu) / sd

    d0 = float(ncfg.get("deadzone", 0.0534))
    if not 0.0 <= d0 < 1.0:
        raise ValueError(f"scoring.u_valid.novel.deadzone は 0 以上 1 未満: {d0}")

    d_b = _nn_min_dist((b[feats].to_numpy(dtype=float) - mu) / sd, Az)
    d_c = _nn_min_dist((c[feats].to_numpy(dtype=float) - mu) / sd, Az)

    d_plus = float(ks_2samp(d_c, d_b, alternative="greater").statistic)
    score = float(np.clip(1.0 - max(0.0, d_plus - d0) / (1.0 - d0), 0.0, 1.0))
    detail = {
        "enabled": True,
        "novel": score,
        "d_plus": d_plus,
        "deadzone": d0,
        "n_rows_C": len(d_c),
        "n_rows_B": len(d_b),
        "nn_dist_C_median": float(np.median(d_c)),
        "nn_dist_B_median": float(np.median(d_b)),
        "n_rows_A_bg": len(A),
        "features": feats,
    }
    return score, detail


def _score_u_valid(
    c: pd.DataFrame, b: pd.DataFrame, ref: UtilityReference, cfg: dict
) -> tuple[float, dict]:
    """内部妥当性 = mean{cens, no_fab, no_dup, novel}。

    - cens（打ち切り整合率）: 打ち切り構造が壊れていないか。**行単位の論理整合 と 母集団の
      打ち切り率のB突き合わせ の両方**を課す（min＝弱いほう。2026-08-20 改修＝
      belt-and-suspenders）。他facetが見逃す1時点横断の論理的不変量（満たすこと=正しいこと・D3）。
    - no_fab（関係捏造の片側罰）: 実データより強い群→指標の関係を作ったCを弱く罰す（D4）。
    - no_dup（完全重複行の罰）: 同一個体の多重計上を弱く罰す。
    - novel（新規性）: A_bg の部分集合を渡すだけの提出を D+ で罰す（§2.1・上の `_score_u_novel`）。
    旧 dir（FPG→発症の向き）/ nondm（非糖尿病FPG率）は U_spec / U_gen と実測collinearで冗長のため
    除外（D1/D2）。生理外の値はソフトでなく schema.ranges のハード棄却で弾く（D6）。集約=mean。

    ★2026-08-20 改修（cens の退化分岐の穴を塞ぐ）: 旧実装は「自分が admin-censored と名乗った
    行の中の time=horizon 一致率」だけを見ており、非発症行に death=1 を立てると admin-censored
    が空になり rate_censor が無条件で 1.0（満点）になった。防御側は time 分布を壊して AIA
    （標的=time）を無効化しつつ cens 満点を保てた。塞ぐために2つのチェックを AND（min）で課す:
      (1) 行単位整合 cens_rowwise: admin-censored 行の time=horizon 一致率。**該当0人は満点でなく
          0.0（異常）**にする（空にできる集合で満点を出さない）。
      (2) 打ち切り率突き合わせ cens_rate_match: 母集団の time=horizon 率を B に合わせる
          `1 − |p_hz(C) − p_hz(B)|`。**母集団の割合なので防御側が空にできない**＝(1) を
          小さな compliant 部分集合だけ残して迂回する手も塞ぐ。
    どちらも C=B・正当な加工（time を保つ加法ノイズ・k-匿名化・合成）では 1.0 のまま
    （恒等アンカーでも |ΔU| = 0.00000）。
    """
    horizon = float(cfg["onset"]["horizon_years"])
    atol = 1e-2

    # (1) 行単位の論理整合。admin-censored（非発症∧非死亡）行の time が horizon か。
    #     該当0人＝「防御側が空にできる集合」なので満点にせず 0.0（異常とみなす）。
    admin_censored = (c["onset"] == 0) & (c["death"] == 0)
    if admin_censored.any():
        cens_rowwise = float(np.isclose(c.loc[admin_censored, "time"], horizon, atol=atol).mean())
    else:
        cens_rowwise = 0.0
    # (2) 母集団の打ち切り率（time=horizon の割合）を B に突き合わせる。割合なので空にできない。
    p_hz_c = float(np.isclose(c["time"].to_numpy(dtype=float), horizon, atol=atol).mean())
    p_hz_b = float(np.isclose(b["time"].to_numpy(dtype=float), horizon, atol=atol).mean())
    cens_rate_match = 1.0 - min(abs(p_hz_c - p_hz_b), 1.0)
    # belt-and-suspenders: 両方を満たすこと（弱いほうを採る）。2026-08-20 cens 穴の修正。
    rate_censor = float(min(cens_rowwise, cens_rate_match))

    no_fab_score, no_fab_detail = _no_fab(c, b, cfg)
    no_dup_score, no_dup_detail = _no_dup(c, cfg)

    parts = [rate_censor, no_fab_score, no_dup_score]
    novel_score, novel_detail = _score_u_novel(c, b, getattr(ref, "a_bg", None), cfg)
    if novel_score is not None:
        parts.append(novel_score)  # A_bg が渡されているときのみ mean-4 に入る

    facet = float(np.mean(parts))
    detail = {
        "censoring_realism_rate": rate_censor,
        "no_fab": no_fab_score,
        "no_fab_detail": no_fab_detail,
        "no_dup": no_dup_score,
        "no_dup_detail": no_dup_detail,
    }
    if novel_score is not None:
        detail["novel"] = novel_score
        detail["novel_detail"] = novel_detail
    return facet, detail


def _aggregate(facets: dict[str, float]) -> float:
    """4観点を **完全 min-4** で合成する。ルールブック §6.1 の公表式。

    `U = min(U_gen, U_spec, U_rare, U_valid)`。4観点を対等に扱い、一番低い観点がそのまま
    有用性になる（得意な観点で苦手を埋め合わせられない）。
    """
    return float(min(facets.values()))


def score_utility(
    C: pd.DataFrame,
    B: pd.DataFrame,
    ref: UtilityReference,
    cfg: dict,
    level: ReportLevel = ReportLevel.AGGREGATE,
) -> UtilityResult:
    """有用性4ファセットを採点する（純関数）。発症曲線 time_dz は U_gen のサブ指標。

    AGGREGATE: score(集約)のみが軽量パス。BREAKDOWN: facets(4ファセット別)に加え、
    各ファセットのサブ指標をdetailに格納（発症曲線サブ指標 time_dz は U_gen 配下）。
    PER_RECORD: 加えて行別のMahalanobis/外れ値検出detailを付す。
    """
    c_static, b_static = C, B
    c = _prepare(C)
    b = _prepare(B)

    time_dz, time_detail = _score_u_time(c, b, ref, cfg)
    gen_score, gen_detail = _score_u_gen(c_static, b_static, time_dz, time_detail, ref, cfg)
    spec_score, spec_detail = _score_u_spec(c, b)
    rare_score, rare_detail = _score_u_rare(c, b, c_static, b_static, ref, cfg)
    valid_score, valid_detail = _score_u_valid(c, b, ref, cfg)

    facets = {
        "U_gen": gen_score,
        "U_spec": spec_score,
        "U_rare": rare_score,
        "U_valid": valid_score,
    }
    score = _aggregate(facets)

    detail: dict | None = None
    if level in (ReportLevel.BREAKDOWN, ReportLevel.PER_RECORD):
        detail = {
            "U_gen": gen_detail,
            "U_spec": spec_detail,
            "U_rare": rare_detail,
            "U_valid": valid_detail,
        }
        if level is ReportLevel.PER_RECORD:
            detail["records"] = {
                "record_id": c["record_id"].to_numpy(),
                "mahalanobis": outlier_mod.mahalanobis(c_static, ref.mu, ref.sigma, schema.CONTINUOUS),
                "is_rare_detected": _rare_mask(c_static, ref),
            }

    return UtilityResult(score=score, facets=facets, detail=detail)
