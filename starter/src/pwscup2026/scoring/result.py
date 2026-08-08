"""採点結果の返り値構造体と report level。

開示ポリシー: 有用性PER_RECORDは防御者に開放（自分のB_i比較＝真値漏洩なし・キット同梱）。
攻撃PER_RECORDは真値参照＝事務局オフライン専用（ライブはAGGREGATEのみ、
プロービング対策）。AGGREGATEでは重いdetailを計算/格納しない（ライブの軽量パス）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class ReportLevel(Enum):
    AGGREGATE = auto()
    BREAKDOWN = auto()
    PER_RECORD = auto()


@dataclass
class UtilityResult:
    score: float
    facets: dict[str, float] = field(default_factory=dict)
    detail: dict | None = None


@dataclass
class AttackResult:
    score: float
    breakdown: dict = field(default_factory=dict)
    detail: dict | None = None


@dataclass
class AnonymityResult:
    score: float
    facets: dict = field(default_factory=dict)
    detail: dict | None = None


@dataclass
class AttackScoreResult:
    """攻撃者1チームの**攻撃得点**（締切後にオフラインで集計）。

    `score` ＝ ファセット横断の攻撃得点（ランキング元）。`breakdown` に素点（mean・ライブ表示相当）と
    ファセット別の得点/素点を保持する。
    """
    score: float
    breakdown: dict = field(default_factory=dict)
    detail: dict | None = None


@dataclass
class RoundScore:
    """1ラウンドのメイン順位得点 = min(有用性, 匿名性)・絶対スケール。
    攻撃得点はメインに畳まず別ランキングにする。"""
    score: float
    breakdown: dict = field(default_factory=dict)


@dataclass
class FinalScore:
    """予備戦×0.1＋本戦×0.9 の最終スカラー。"""
    score: float
    breakdown: dict = field(default_factory=dict)


def json_safe(obj):
    """`json.dumps` が **正しい JSON** を書けるように、非有限の float を `None`（null）に置き換える。

    ★2026-08-01 追加。`attack_score.score_mia` は希少層に陽性が居ないとき
    `breakdown["tpr_rare"]` を `float("nan")` にする（退化安全のための設計どおりの値）。
    ところが Python の `json.dumps` は既定でこれを `NaN` という**JSON 仕様に無いリテラル**として
    書き出すため、CodaBench の詳細結果（`detailed_results.html` に貼る内訳）や
    `scores.json` が、JSON パーサから見ると壊れたファイルになっていた。
    参加者から見ると「棄却理由を確認する唯一の窓口」が壊れて見える。

    NaN のまま出さず、値が無いことを `null` で表す（`nan` を 0 に潰すと
    「希少層で TPR が 0 だった」と読めてしまい、意味が変わる）。
    """
    import math

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
