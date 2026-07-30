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
    """

    mu: np.ndarray
    sigma: np.ndarray
    band: dict[str, float]
    b_is_rare: np.ndarray | None = None
    km_signal_D: float | None = None
    km_floor: float | None = None


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


def _score_u_gen(
    c_static: pd.DataFrame, b_static: pd.DataFrame, time_dz: float, time_detail: dict
) -> tuple[float, dict]:
    """一般有用性＝min(marginal_mean, pmse_score, time_dz) の弱点支配集約。

    3指標は直交する相補的な故障検出器（周辺分布 / joint忠実度pMSE / 発症曲線）ゆえ min で集約。
    marginal_mean=10列(連続7=1−KS / カテゴリ3=1−TVD)の平均。旧式は marginal_mean と marginal_worst を
    共に平均へ入れ周辺を二重計上していたが、min では marginal_mean のみ採用（marginal は worst でなく
    mean＝k-anon を DP 並みに過剰減点しないため）。marginal_worst はスコアから外し診断 detail にのみ残す
    （ダッシュボード/オフライン集計は BREAKDOWN で各サブ指標を取り出せる）。time_dz は旧 U_time（発症フリー
    KM曲線L1のデッドゾーン正規化）を統合した安全網＝通常
    1.0、発症構造破壊時のみ沈む。
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

    facet = float(min(marginal_mean, pmse_score, time_dz))  # min-3（worstはmeanに畳まず診断のみ）
    detail = {
        "marginal": marginal,
        "marginal_mean": marginal_mean,
        "marginal_worst": marginal_worst,
        "pmse": pmse,
        "pmse_ceiling": ceiling,
        "pmse_score": pmse_score,
        "time_dz": time_dz,
        "time": time_detail,
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


def _c2st_raw(
    x_rare: pd.DataFrame, y_rare: pd.DataFrame, markers: list[str], min_rare_n: int, n_estimators: int
) -> float | None:
    """希少群の**joint忠実度**＝2標本分類テスト（C2ST）。マーカー空間で x_rare(=C検出希少) と
    y_rare(=B真希少) をRandomForestで判別し、`1 - 2|AUC-0.5|`（判別不能=1=joint一致）を返す。
    prof(1-KS=周辺のみ)が見落とす**同時分布/裾依存**を捉える（copulaのガウス依存が苦手な軸）。
    （希少層の同時分布まで見る軸）。

    決定性: RF/CV とも random_state=0、かつ **RF は n_jobs=1**（並列数で加算順が変わり
    機械をまたぐと結果がぶれるため。下のコメント参照）。データ僅少（n_splits<2）なら
    None（較正で除外）。
    """
    if len(x_rare) < min_rare_n or len(y_rare) < min_rare_n:
        return None
    X = np.vstack([x_rare[markers].to_numpy(float), y_rare[markers].to_numpy(float)])
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
        参照側が不足（世界に希少が少ない）なら従来どおり除外(None)。それ以外は自己ベースライン較正。
        旧実装は C側不足でも除外(None)だったため、検出希少を8未満に絞ると c2st が min から外れて
        U_rare が水増しできた（dodge）。"""
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


def _score_u_valid(c: pd.DataFrame, b: pd.DataFrame, cfg: dict) -> tuple[float, dict]:
    """内部妥当性 = mean{cens, no_fab}。

    - cens（打ち切り妥当率）: 非発症∧非死亡→time=horizon の一致率。他facetが見逃す1時点横断の
      論理的不変量（このスキーマ唯一の強い関係的論理不変量・満たすこと=正しいこと・D3）。
    - no_fab（関係捏造の片側罰）: 実データより強い群→指標の関係を作ったCを弱く罰す（D4）。
    旧 dir（FPG→発症の向き）/ nondm（非糖尿病FPG率）は U_spec / U_gen と実測collinearで冗長のため
    除外（D1/D2）。生理外の値はソフトでなく schema.ranges のハード棄却で弾く（D6）。集約=mean。
    """
    horizon = float(cfg["onset"]["horizon_years"])

    admin_censored = (c["onset"] == 0) & (c["death"] == 0)
    if admin_censored.any():
        rate_censor = float(np.isclose(c.loc[admin_censored, "time"], horizon, atol=1e-2).mean())
    else:
        rate_censor = 1.0

    no_fab_score, no_fab_detail = _no_fab(c, b, cfg)

    facet = float(np.mean([rate_censor, no_fab_score]))
    detail = {
        "censoring_realism_rate": rate_censor,
        "no_fab": no_fab_score,
        "no_fab_detail": no_fab_detail,
    }
    return facet, detail


def _aggregate(facets: dict[str, float], cfg: dict) -> float:
    """facetsを合成する。`scoring.utility_aggregation` で方式選択。

    - "min"(現行): **完全min-4** `U=min(U_gen,U_spec,U_rare,U_valid)`。
      4観点を対等に扱い、一番低い観点がそのまま有用性になる（得意な観点で苦手を埋め合わせられない）。
      ルールブック §6.1 の公表式。
    - "mean"(旧): `utility_weights` による加重平均（null=等重み）。
    - "rare_gate"(旧): U_rare を独立軸に昇格し score = min(U_rare, core)。
      core = `utility_core_weights` による {U_gen,U_spec,U_valid} の加重平均（U_valid は識別力最小・
      飽和のため軽く＝既定0.5）。round_score の min(score, A) と合わせ **main = min(U_rare, core, A)**
      ＝「希少を取れるか」を勝敗の鍵に。stress-testで、A(aia)束縛下ではU内の重み変更は
      不可視（ρ=1.000）で、min独立軸化のみがU_rareを効かせると実測。
    """
    mode = cfg["scoring"].get("utility_aggregation", "mean")
    if mode == "min":
        return float(min(facets.values()))
    if mode == "rare_gate":
        rare = facets["U_rare"]
        core_facets = {k: v for k, v in facets.items() if k != "U_rare"}
        cw = cfg["scoring"].get("utility_core_weights") or {}
        total_w = sum(cw.get(k, 0.0) for k in core_facets)
        if total_w > 0:
            core = float(sum(core_facets[k] * cw.get(k, 0.0) for k in core_facets) / total_w)
        else:
            core = float(np.mean(list(core_facets.values())))
        return float(min(rare, core))
    weights = cfg["scoring"].get("utility_weights")
    if not weights:
        return float(np.mean(list(facets.values())))
    total_w = sum(weights.get(k, 0.0) for k in facets)
    if total_w <= 0:
        return float(np.mean(list(facets.values())))
    return float(sum(facets[k] * weights.get(k, 0.0) for k in facets) / total_w)


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
    gen_score, gen_detail = _score_u_gen(c_static, b_static, time_dz, time_detail)
    spec_score, spec_detail = _score_u_spec(c, b)
    rare_score, rare_detail = _score_u_rare(c, b, c_static, b_static, ref, cfg)
    valid_score, valid_detail = _score_u_valid(c, b, cfg)

    facets = {
        "U_gen": gen_score,
        "U_spec": spec_score,
        "U_rare": rare_score,
        "U_valid": valid_score,
    }
    score = _aggregate(facets, cfg)

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
