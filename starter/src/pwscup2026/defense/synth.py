"""汎用合成器 `copula_synth` / `copula_survival`（B_i のみから学習）。

B_i の配布形式だけから周辺（経験分布）＋依存（ガウスコピュラ）を独立に再学習して
n行再サンプルする、参加者が現実に書ける水準の generic synthesizer。真値を覗かないよう、
データ生成側の実装は一切再利用せず numpy/scipy/pandas だけで完結させている。

- `copula_synth`    : 共変量=コピュラ再合成、**転帰は共変量と独立に**再サンプル。
- `copula_survival` : 共変量は同じ、**転帰を cause-specific Cox（競合リスク）で共変量条件付きに**
  サンプルする。`copula_synth` は転帰の紐付けを作らないため U_spec（特定解析での結論一致）が
  立たない。その穴を
  塞いだ版で、追加依存は lifelines のみ。**スターターキットの参照防御はこちら**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..common import schema, submission_schema

_TIME_FLOOR_YEARS = 1.0 / 365.25
_STATE_ORDER = ((0, 0), (1, 0), (0, 1))  # (onset, death) の許容3状態
_UNIFORM_EPS = 1e-6
#: `copula_survival` の cause-specific Cox 予測子＝9共変量（sex/smoking は 0/1）。採点側の Cox と同じ集合。
_COX_PREDICTORS = list(schema.COVARIATES)


def _nearest_correlation(corr: np.ndarray) -> np.ndarray:
    """固有値クリップで半正定値の相関行列に丸める（定数列でのnan混入対策込み）。"""
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    fixed = eigvecs @ np.diag(np.clip(eigvals, 1e-8, None)) @ eigvecs.T
    scale = np.sqrt(np.diag(fixed))
    scale = np.where(scale > 0, scale, 1.0)
    fixed = fixed / np.outer(scale, scale)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def _pseudo_observations(x: np.ndarray) -> np.ndarray:
    """列ごと順位変換で擬似一様観測u∈(0,1)を作る（タイは平均順位）。"""
    n = x.shape[0]
    ranks = np.apply_along_axis(lambda col: pd.Series(col).rank(method="average").to_numpy(), 0, x)
    return ranks / (n + 1)


def _resample_continuous(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """経験分位→ガウスコピュラfit→同数サンプル→経験CDF逆変換で連続列を再構成（手順1）。"""
    cols = schema.CONTINUOUS
    x = df[cols].to_numpy(dtype=float)
    n = len(df)

    u = np.clip(_pseudo_observations(x), _UNIFORM_EPS, 1.0 - _UNIFORM_EPS)
    z = norm.ppf(u)
    corr = _nearest_correlation(np.corrcoef(z, rowvar=False))

    z_samp = rng.multivariate_normal(mean=np.zeros(len(cols)), cov=corr, size=n)
    u_samp = np.clip(norm.cdf(z_samp), 0.0, 1.0)

    out = np.empty_like(x)
    for i in range(len(cols)):
        sorted_vals = np.sort(x[:, i])
        out[:, i] = np.quantile(sorted_vals, u_samp[:, i], method="linear")
    return pd.DataFrame(out, columns=cols)


def _resample_categorical(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """sex/smoking/prefectureを周辺独立にmultinomial再サンプル（手順2）。"""
    n = len(df)
    out: dict[str, np.ndarray] = {}
    for col in ("sex", "smoking"):
        vals, counts = np.unique(df[col].to_numpy(dtype=int), return_counts=True)
        p = counts / counts.sum()
        out[col] = rng.choice(vals, size=n, p=p).astype(int)

    pref_vals, pref_counts = np.unique(df["prefecture"].to_numpy(dtype=object), return_counts=True)
    p_pref = pref_counts / pref_counts.sum()
    out["prefecture"] = rng.choice(pref_vals, size=n, p=p_pref)
    return out


def _resample_outcome(df: pd.DataFrame, horizon: float, rng: np.random.Generator) -> pd.DataFrame:
    """(onset,death)3状態を周辺頻度から、timeは状態内経験分布から再サンプル（手順3）。"""
    n = len(df)
    onset = df["onset"].to_numpy(dtype=int)
    death = df["death"].to_numpy(dtype=int)
    time = df["time"].to_numpy(dtype=float)
    states = list(zip(onset.tolist(), death.tolist()))

    state_masks = {s: np.array([st == s for st in states]) for s in _STATE_ORDER}
    probs = np.array([state_masks[s].sum() / n for s in _STATE_ORDER])

    draw_idx = rng.choice(len(_STATE_ORDER), size=n, p=probs)
    out_onset = np.empty(n, dtype=int)
    out_death = np.empty(n, dtype=int)
    out_time = np.empty(n, dtype=float)
    for k, s in enumerate(_STATE_ORDER):
        mask = draw_idx == k
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        out_onset[mask] = s[0]
        out_death[mask] = s[1]
        pool = time[state_masks[s]]
        out_time[mask] = rng.choice(pool, size=cnt, replace=True)

    out_time = np.clip(out_time, _TIME_FLOOR_YEARS, horizon)
    return pd.DataFrame({"time": out_time, "onset": out_onset, "death": out_death})


def copula_synth(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """B_iだけから周辺+依存を独立に再学習して同数再合成する。"""
    record_id = df["record_id"].to_numpy()

    continuous = _resample_continuous(df, rng)
    categorical = _resample_categorical(df, rng)

    out = pd.DataFrame({"record_id": record_id})
    for col in schema.CONTINUOUS:
        out[col] = continuous[col].to_numpy()
    for col in ("sex", "smoking", "prefecture"):
        out[col] = categorical[col]

    age_lo, age_hi = (float(v) for v in cfg["population"]["age_range"])
    age = np.clip(out["age"].to_numpy(dtype=float), age_lo, age_hi)
    out["age"] = np.floor(age).astype(int)

    ranges = cfg["schema"]["ranges"]
    for col in ("BMI", "SBP", "TG", "HDL", "ALT", "FPG"):
        lo, hi = (float(v) for v in ranges[col])
        out[col] = np.clip(out[col].to_numpy(dtype=float), lo, hi)

    horizon = float(cfg["onset"]["horizon_years"])
    outcome_out = _resample_outcome(df, horizon, rng)
    out["time"] = outcome_out["time"].to_numpy()
    out["onset"] = outcome_out["onset"].to_numpy()
    out["death"] = outcome_out["death"].to_numpy()

    return out[submission_schema.DISTRIBUTED_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# copula_survival: 共変量=コピュラ／転帰=cause-specific Cox の条件付きサンプル
# --------------------------------------------------------------------------- #
def _fit_cause_cox(df: pd.DataFrame, event_col: str) -> tuple[object, np.ndarray, np.ndarray]:
    """原因別（cause-specific）Coxを B_i にfitし、(model, 基準累積ハザードの時刻, 値) を返す。

    競合リスク＝発症と死亡をそれぞれ「相手をcensor扱い」でfitする。
    ここは**防御側の合成器**であって採点ではないので、収束優先で軽い penalizer を入れる
    （採点の Cox は素MLE。両者は別物）。
    """
    from lifelines import CoxPHFitter

    d = df[_COX_PREDICTORS + ["time"]].copy()
    d["_event"] = df[event_col].to_numpy(dtype=int)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(d, duration_col="time", event_col="_event", robust=False)
    h0 = cph.baseline_cumulative_hazard_.iloc[:, 0]
    return cph, h0.index.to_numpy(dtype=float), h0.to_numpy(dtype=float)


def _sample_event_time(
    partial_hazard: np.ndarray, times: np.ndarray, h0_values: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """H0(t)·ph = −ln(U) を満たす t を H0 の逆関数（線形補間）で引く。範囲外は inf＝打ち切り。"""
    target = -np.log(np.clip(u, 1e-12, 1.0)) / np.clip(partial_hazard, 1e-12, None)
    t = np.interp(target, h0_values, times, left=times[0], right=np.inf)
    return np.where(target > h0_values[-1], np.inf, t)


def copula_survival(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """共変量＝コピュラ再合成、転帰＝cause-specific Cox で共変量条件付きにサンプルする。

    `copula_synth` との違いは転帰だけ。`copula_synth` は転帰を共変量と独立に引くので
    「FPG→発症」のような関係が合成データに残らず、U_spec（特定解析での結論一致）が立たない。
    こちらは B_i にfitした競合リスクCoxから発症時刻・死亡時刻を引いて先に起きた方を採用するため、
    共変量と転帰の関係が保たれる。B_i の配布形式だけを使う（真値は覗かない）。
    """
    n = len(df)
    horizon = float(cfg["onset"]["horizon_years"])

    continuous = _resample_continuous(df, rng)
    categorical = _resample_categorical(df, rng)
    out = pd.DataFrame({"record_id": df["record_id"].to_numpy()})
    for col in schema.CONTINUOUS:
        out[col] = continuous[col].to_numpy()
    for col in ("sex", "smoking", "prefecture"):
        out[col] = categorical[col]

    age_lo, age_hi = (float(v) for v in cfg["population"]["age_range"])
    out["age"] = np.floor(np.clip(out["age"].to_numpy(dtype=float), age_lo, age_hi)).astype(int)
    ranges = cfg["schema"]["ranges"]
    for col in ("BMI", "SBP", "TG", "HDL", "ALT", "FPG"):
        lo, hi = (float(v) for v in ranges[col])
        out[col] = np.clip(out[col].to_numpy(dtype=float), lo, hi)

    cph_onset, t_onset, h0_onset = _fit_cause_cox(df, "onset")
    cph_death, t_death, h0_death = _fit_cause_cox(df, "death")
    x = out[_COX_PREDICTORS].astype(float)
    time_onset = _sample_event_time(
        cph_onset.predict_partial_hazard(x).to_numpy(), t_onset, h0_onset, rng.uniform(size=n)
    )
    time_death = _sample_event_time(
        cph_death.predict_partial_hazard(x).to_numpy(), t_death, h0_death, rng.uniform(size=n)
    )

    # 先に起きたイベントを採用。どちらも地平より後なら行政的右打ち切り（(onset,death)=(0,0)・time=horizon）。
    censored = np.minimum(time_onset, time_death) > horizon
    onset_event = (~censored) & (time_onset <= time_death)
    death_event = (~censored) & (time_death < time_onset)

    onset = np.zeros(n, dtype=int)
    death = np.zeros(n, dtype=int)
    time = np.full(n, horizon, dtype=float)
    onset[onset_event] = 1
    time[onset_event] = time_onset[onset_event]
    death[death_event] = 1
    time[death_event] = time_death[death_event]

    out["time"] = np.clip(time, _TIME_FLOOR_YEARS, horizon)  # time>0 と地平を保証（値域reject回避）
    out["onset"] = onset
    out["death"] = death  # onset_event と death_event は排他なので (1,1) は出ない
    return out[submission_schema.DISTRIBUTED_COLUMNS].reset_index(drop=True)
