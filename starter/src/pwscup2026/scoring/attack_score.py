"""攻撃力の採点。

通貨＝TPR@low-FPR（Carlini）。MIAはchance基準（TPR@low-FPR自体がFPR=chance線基準）、
"""
from __future__ import annotations

import numpy as np

from .result import AttackResult, ReportLevel


def _tpr_at_fpr(confidence: np.ndarray, truth: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """ROC点を作りtarget_fprでのTPRを線形補間する。AUCも返す。陽性/陰性いずれか無しは退化扱い。

    **同じ確信度の行はひとつの動作点にまとめる**（`sklearn.roc_curve` と同じ）。まとめないと
    同値の塊の中の並び順＝提出CSVの行順が結果を左右し、全行に同じ値を出した提出の TPR が
    行の並べ方次第で 0 にも 1 にもなってしまう。ルールブック §6.2.1 が約束している
    「同じ値を出すと ROC が対角線になり、FPR=0.01 の TPR も偶然の水準になる」を成立させるため、
    ここで同値をまとめる。
    """
    order = np.argsort(-confidence, kind="mergesort")
    c = confidence[order]
    y = truth[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.5

    idx = np.r_[np.flatnonzero(np.diff(c) != 0), len(c) - 1]  # 同値塊の末尾だけを動作点にする
    tp = np.cumsum(y)[idx]
    fp = np.cumsum(1 - y)[idx]
    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / n_neg]
    tpr_at = float(np.interp(target_fpr, fpr, tpr))
    auc = float(np.trapezoid(tpr, fpr))
    return tpr_at, auc


def score_mia(
    confidence: np.ndarray,
    truth_membership: np.ndarray,
    cfg: dict,
    level: ReportLevel = ReportLevel.AGGREGATE,
    rare_mask: np.ndarray | None = None,
) -> AttackResult:
    """overlap-MIAの確信度をTPR@low-FPRで採点する。

    confidence: 候補行ごと[0,1]。truth_membership: O_ij由来のメンバー0/1
    （truth_mapでrecord_id→pool_id突合済みのものを渡す）。

    `rare_mask`（候補行と同順・True=希少）を渡すと**希少層 worst-case**を取る:
    全体プールのTPR@low-FPRと、希少候補のみに絞った同動作点のTPRの max（最も露出した層）。
    希少層に陽性が居ない/mask空なら全体プールのみ（退化安全）。
    """
    confidence = np.asarray(confidence, dtype=float)
    truth = np.asarray(truth_membership, dtype=int)
    low_fpr = float(cfg["scoring"]["low_fpr"])
    tpr, auc = _tpr_at_fpr(confidence, truth, low_fpr)

    tpr_rare = float("nan")
    score = tpr
    if rare_mask is not None:
        rare_mask = np.asarray(rare_mask, dtype=bool)
        if rare_mask.any() and int(truth[rare_mask].sum()) > 0:
            tpr_rare, _ = _tpr_at_fpr(confidence[rare_mask], truth[rare_mask], low_fpr)
            score = max(tpr, tpr_rare)

    breakdown = {"tpr_at_low_fpr": tpr, "tpr_rare": tpr_rare, "score": score, "low_fpr": low_fpr, "auc": auc}
    detail = None
    if level is ReportLevel.PER_RECORD:
        detail = {"confidence": confidence, "truth_membership": truth, "rare_mask": rare_mask}
    return AttackResult(score=score, breakdown=breakdown, detail=detail)


def R_from_rate(succ_m: np.ndarray, succ_c: np.ndarray) -> float:
    """会員/対照の成功率0/1配列から正規化超過を返す。

    `clip((mean(succ_m) - mean(succ_c)) / max(1 - mean(succ_c), 1e-9), 0, 1)`。
    どちらかが空ならnan。
    """
    succ_m = np.asarray(succ_m, dtype=float)
    succ_c = np.asarray(succ_c, dtype=float)
    if len(succ_m) == 0 or len(succ_c) == 0:
        return float("nan")
    p_m = float(np.mean(succ_m))
    p_c = float(np.mean(succ_c))
    return float(np.clip((p_m - p_c) / max(1.0 - p_c, 1e-9), 0.0, 1.0))


def R_from_rate_deadzone(succ_m: np.ndarray, succ_c: np.ndarray, z: float) -> float:
    """`R_from_rate` に片側 z·SE のデッドゾーンを入れた版（希少 worst-case 用）。

    `clip((p_m − p_c − z·SE)/(1 − p_c), 0, 1)`、`SE = sqrt(p_m(1−p_m)/n_m + p_c(1−p_c)/n_c)`。
    希少層は標本が小さく、偶然の差で防御者を罰してしまうため、
    統計的な誤差の範囲に収まる差は無罰にする（`U_rare.mass_star` の SE デッドゾーンと同流儀）。
    z=0 で生の `R_from_rate` と同じ挙動になる。
    """
    succ_m = np.asarray(succ_m, dtype=float)
    succ_c = np.asarray(succ_c, dtype=float)
    if len(succ_m) == 0 or len(succ_c) == 0:
        return float("nan")
    if z <= 0.0:
        return R_from_rate(succ_m, succ_c)
    p_m = float(np.mean(succ_m))
    p_c = float(np.mean(succ_c))
    se = float(np.sqrt(p_m * (1.0 - p_m) / len(succ_m) + p_c * (1.0 - p_c) / len(succ_c)))
    return float(np.clip((p_m - p_c - z * se) / max(1.0 - p_c, 1e-9), 0.0, 1.0))


def R_from_tpr(
    conf_m: np.ndarray, truth_m: np.ndarray, conf_c: np.ndarray, truth_c: np.ndarray, low_fpr: float
) -> tuple[float, float, float]:
    """会員/対照それぞれのTPR@low-FPRを取り、`R_from_rate`と同じ正規化超過を返す。

    Returns: (R, tpr_member, tpr_control)。
    """
    tpr_m, _ = _tpr_at_fpr(np.asarray(conf_m, dtype=float), np.asarray(truth_m, dtype=int), low_fpr)
    tpr_c, _ = _tpr_at_fpr(np.asarray(conf_c, dtype=float), np.asarray(truth_c, dtype=int), low_fpr)
    r = float(np.clip((tpr_m - tpr_c) / max(1.0 - tpr_c, 1e-9), 0.0, 1.0))
    return r, tpr_m, tpr_c


def score_aia(
    member_onset_conf: np.ndarray | None,
    member_time_hat: np.ndarray,
    member_truth_onset: np.ndarray,
    member_truth_time: np.ndarray,
    control_onset_conf: np.ndarray | None,
    control_time_hat: np.ndarray,
    control_truth_onset: np.ndarray,
    control_truth_time: np.ndarray,
    cfg: dict,
    level: ReportLevel = ReportLevel.AGGREGATE,
    member_rare_mask: np.ndarray | None = None,
    control_rare_mask: np.ndarray | None = None,
) -> AttackResult:
    """aiaの確信度を会員/対照間の正規化超過で採点する。

    R_time（本命・主score）: 成功=`|time_hat-time_truth| <= aia.tau_time`（年）
    の会員/対照間`R_from_rate`。R_onset（副・breakdownのみ）: onset_confのTPR@low-FPR
    （`scoring.low_fpr`）の会員/対照間`R_from_tpr`（不均衡でノイジー＝副次）。

    `member_rare_mask`/`control_rare_mask`（各群と同順・True=希少）を両方渡すと
    **希少層 worst-case**を取る: 全体の R_time と、希少会員/希少対照に絞った R_time の max。
    どちらか空なら全体のみ（退化安全）。

    希少側だけは `scoring.aia_rare_se_z`（既定0＝旧挙動・本番1.64）の z·SE デッドゾーンを通す
    （13人標本の偶然差で罰しないため。全体 R_time には入れない＝
    SEが大きく中間帯の漏洩まで消えるため）。
    """
    tau = float(cfg["aia"]["tau_time"])
    low_fpr = float(cfg["scoring"]["low_fpr"])
    rare_z = float(cfg["scoring"].get("aia_rare_se_z", 0.0))

    succ_m = np.abs(np.asarray(member_time_hat, dtype=float) - np.asarray(member_truth_time, dtype=float)) <= tau
    succ_c = np.abs(np.asarray(control_time_hat, dtype=float) - np.asarray(control_truth_time, dtype=float)) <= tau
    r_time_all = R_from_rate(succ_m.astype(float), succ_c.astype(float))

    r_time = r_time_all
    r_time_rare = float("nan")
    r_time_rare_raw = float("nan")
    succ_m_rare = float("nan")
    succ_c_rare = float("nan")
    if member_rare_mask is not None and control_rare_mask is not None:
        mm = np.asarray(member_rare_mask, dtype=bool)
        cm = np.asarray(control_rare_mask, dtype=bool)
        if mm.any() and cm.any():
            # 希少側の生の成功率（p_m/p_c）。R_from_rate_deadzone が内部で使う値と同じで、
            # 詳細結果に「どれだけ当てて、対照はどれだけ当たってしまったか」を出すために取り出す。
            succ_m_rare = float(np.mean(succ_m[mm]))
            succ_c_rare = float(np.mean(succ_c[cm]))
            r_time_rare_raw = R_from_rate(succ_m[mm].astype(float), succ_c[cm].astype(float))
            r_time_rare = R_from_rate_deadzone(
                succ_m[mm].astype(float), succ_c[cm].astype(float), rare_z
            )
            if not np.isnan(r_time_rare):
                r_time = max(r_time_all, r_time_rare)

    # onset側は診断のみ（採点は R_time）。提出物からは onset_conf を廃止したので None で来る。
    if member_onset_conf is None or control_onset_conf is None:
        r_onset = tpr_m = tpr_c = float("nan")
    else:
        r_onset, tpr_m, tpr_c = R_from_tpr(
            member_onset_conf, member_truth_onset, control_onset_conf, control_truth_onset, low_fpr
        )

    breakdown = {
        "r_onset": r_onset,
        "r_time_all": r_time_all,
        "r_time_rare": r_time_rare,          # デッドゾーン適用後（採点に入る値）
        "r_time_rare_raw": r_time_rare_raw,  # 生値（診断・デッドゾーンの効きを見る）
        "aia_rare_se_z": rare_z,
        "tpr_onset_member": tpr_m,
        "tpr_onset_control": tpr_c,
        "succ_time_member": float(np.mean(succ_m)) if len(succ_m) else float("nan"),
        "succ_time_control": float(np.mean(succ_c)) if len(succ_c) else float("nan"),
        "succ_time_member_rare": succ_m_rare,    # 希少会員の成功率（希少マスク未指定/空ならnan）
        "succ_time_control_rare": succ_c_rare,   # 希少対照の成功率（同上）
        "tau_time": tau,
        "low_fpr": low_fpr,
    }
    detail = None
    if level is ReportLevel.PER_RECORD:
        detail = {
            "member_time_hat": member_time_hat,
            "member_truth_time": member_truth_time,
            "control_time_hat": control_time_hat,
            "control_truth_time": control_truth_time,
            "member_onset_conf": member_onset_conf,
            "control_onset_conf": control_onset_conf,
        }
    return AttackResult(score=r_time, breakdown=breakdown, detail=detail)
