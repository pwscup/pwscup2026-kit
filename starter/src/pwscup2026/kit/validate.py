"""提出物フォーマット検証器（スターターキット同梱）。

2層:
- **層1 packaging**（CodaBenchが門前払いする面）: zip健全性 / 直下フラット（フォルダ不可）/
  必須ファイル過不足 / サイズ上限 / 各CSVがUTF-8でparse可・非空 / `token.txt`形式。
- **層2 logical**（提出種別）: 防御=`validate_c_submission`（13列・値域・行数・排他）、
  攻撃=MIA密行列（record_id行×ターゲット列・[0,1]）＋AIAロング（列固定・確信度[0,1]・time≥0）。

純関数＋薄I/Oアダプタ。同一コードを starter kit 自己チェックと将来のCodaBenchライブrejectで使う。
層1で落ちたら層2は走らせない（列が壊れた状態の値チェックはノイズ）。

CLI:
    python -m pwscup2026.kit.validate <submission.zip> [--kind auto|defense|attack] [--dist <配布物ディレクトリ>]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import config as config_mod
from ..common import schema as schema_mod
from ..common import submission_schema
from ..common.submission_schema import ValidationResult
from ..rare import outlier as outlier_mod

#: パッケージ同梱の採点・検証パラメータ（採点と検証が読む部分だけを抜いたもの）。
#: CLI の `--config` の既定値。
KIT_CONFIG_PATH = Path(__file__).with_name("kit_config.yaml")

#: 提出種ごとの必須ファイル集合（過不足ともreject）。
DEFENSE_FILES = {"C.csv", "token.txt"}
ATTACK_FILES = {"F_mia.csv", "F_aia.csv", "token.txt"}
#: AIAロングの固定列。
AIA_COLUMNS = ["target_k", "challenge_row_id", "time_hat"]
#: CodaBench が採点実行時に res dir へ注入する非提出ファイル。層1のディレクトリ検査では
#: これらを無視する（提出物でないため「余計なファイル」に数えない）。
_CODABENCH_INJECTED = {"metadata"}


@dataclass
class PackageResult:
    """層1の返り値。`frames`=parse済CSV（層2へ渡す）、`kind`=defense|attack、`token`=形式検証済トークン。"""

    ok: bool
    errors: list[str] = field(default_factory=list)
    kind: str | None = None
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    token: str | None = None


def _mb(n_bytes: float) -> float:
    return n_bytes / (1024.0 * 1024.0)


def _detect_kind(names: set[str]) -> str | None:
    if "C.csv" in names:
        return "defense"
    if "F_mia.csv" in names or "F_aia.csv" in names:
        return "attack"
    return None


# --------------------------------------------------------------------------- #
# 層1: packaging
# --------------------------------------------------------------------------- #
def check_package(zip_path: str | Path, cfg: dict) -> PackageResult:
    """提出zipのパッケージ整合を検証する。純関数（読み取りのみ・副作用なし）。"""
    errors: list[str] = []
    zp = Path(zip_path)
    scfg = cfg["submission"]
    max_zip, max_unzip = float(scfg["max_zip_mb"]), float(scfg["max_unzipped_mb"])
    token_pat = scfg["token_pattern"]

    if not zp.exists():
        return PackageResult(False, [f"zipが存在しない: {zp}"])
    if _mb(zp.stat().st_size) > max_zip:
        errors.append(f"zipサイズ上限超過: {_mb(zp.stat().st_size):.1f}MB > {max_zip}MB")
    if not zipfile.is_zipfile(zp):
        errors.append("zipとして開けない（壊れている/zipでない）")
        return PackageResult(False, errors)

    with zipfile.ZipFile(zp) as z:
        infos = z.infolist()
        flat_names: list[str] = []
        for info in infos:
            name = info.filename
            if info.is_dir() or name.endswith("/"):
                errors.append(f"ディレクトリ混入（直下フラットでない）: {name}")
            elif "/" in name:
                if name.startswith("__MACOSX/") or name.rsplit("/", 1)[-1].startswith("."):
                    errors.append(f"不要ファイル混入: {name}")
                else:
                    errors.append(f"サブフォルダ混入（直下フラットでない）: {name}")
            elif name.startswith("."):
                errors.append(f"隠しファイル混入: {name}")
            else:
                flat_names.append(name)

        if _mb(sum(i.file_size for i in infos)) > max_unzip:
            errors.append(f"解凍後サイズ上限超過: {_mb(sum(i.file_size for i in infos)):.1f}MB > {max_unzip}MB")

        names = set(flat_names)
        kind = _detect_kind(names)
        if kind is None:
            errors.append(f"提出種を判定できない（C.csv / F_mia.csv+F_aia.csv のいずれも直下に無い）: {sorted(names)}")
            return PackageResult(False, errors)

        required = DEFENSE_FILES if kind == "defense" else ATTACK_FILES
        missing, extra = required - names, names - required
        if missing:
            errors.append(f"必須ファイル不足: {sorted(missing)}")
        if extra:
            errors.append(f"余計なファイル: {sorted(extra)}")

        frames: dict[str, pd.DataFrame] = {}
        for name in sorted(names & required):
            if not name.endswith(".csv"):
                continue
            raw = z.read(name)
            if len(raw) == 0:
                errors.append(f"{name}: 空ファイル")
                continue
            try:
                df = submission_schema.read_scoring_csv(io.BytesIO(raw), encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"{name}: UTF-8でデコードできない")
                continue
            except Exception as e:  # noqa: BLE001 — parse失敗は種別問わずreject
                errors.append(f"{name}: CSVとしてparseできない（{type(e).__name__}）")
                continue
            if df.shape[0] == 0:
                errors.append(f"{name}: データ行が無い")
            frames[name] = df

        token = None
        if "token.txt" in names:
            try:
                raw_txt = z.read("token.txt").decode("utf-8")
            except UnicodeDecodeError:
                errors.append("token.txt: UTF-8でデコードできない")
            else:
                lines = [ln for ln in raw_txt.splitlines() if ln.strip()]
                if len(lines) != 1:
                    errors.append(f"token.txt: 1行でない（非空{len(lines)}行）")
                else:
                    token = lines[0].strip()
                    if not re.match(token_pat, token):
                        errors.append(f"token.txt: 形式不一致（{token_pat}）")

    return PackageResult(len(errors) == 0, errors, kind, frames, token)


# --------------------------------------------------------------------------- #
# 層2: logical（攻撃）
# --------------------------------------------------------------------------- #
def _check_unit_interval(df: pd.DataFrame, cols: list[str], label: str, errors: list[str]) -> None:
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if np.isnan(x).any():
            errors.append(f"{label}[{c}]: 数値化できない/欠測がある")
        bad = np.isfinite(x) & ((x < 0.0) | (x > 1.0))
        if bad.any():
            errors.append(f"{label}[{c}]: 確信度[0,1]外が{int(bad.sum())}個")


def validate_mia_submission(df: pd.DataFrame, cfg: dict, dist: dict | None = None) -> ValidationResult:
    """MIA密行列（行=自B_jのrecord_id × 列=各target_k × セル∈[0,1]）。"""
    errors: list[str] = []
    if "record_id" not in df.columns:
        return ValidationResult(False, ["F_mia: record_id列が無い"])
    rid = pd.to_numeric(df["record_id"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(rid).any() or not np.allclose(rid, np.round(rid)):
        errors.append("F_mia: record_idが整数でない/欠測がある")
    if df["record_id"].duplicated().any():
        errors.append("F_mia: record_id重複（行キー一意違反）")

    target_cols = [c for c in df.columns if c != "record_id"]
    if not target_cols:
        errors.append("F_mia: ターゲット列が1つも無い")
    non_int = [c for c in target_cols if not re.fullmatch(r"\d+", str(c))]
    if non_int:
        errors.append(f"F_mia: 整数でない列ラベル: {non_int}")
    if dist and "mia_columns" in dist:
        allowed = set(dist["mia_columns"]["target_columns"])
        unknown = [c for c in target_cols if str(c) not in allowed]
        if unknown:
            errors.append(f"F_mia: 未知のターゲット列: {unknown}")
        # ★過不足の「不足」側: 自チーム分の1列だけは欠けてよい（自己攻撃はしない）。
        missing = sorted(allowed - {str(c) for c in target_cols}, key=lambda x: int(x))
        own = dist.get("team_id")
        if own is not None:
            # ★提出者が分かる本番では「欠けてよいのは自チームの列だけ」を字面どおりに見る。
            #   「1列までなら何が欠けてもよい」だと、攻撃対象でない攻撃者や自列を入れた提出が、
            #   他チームの列を1つ落としても通ってしまう（ルールブック §5.2）。
            others = [c for c in missing if int(c) != int(own)]
            if others:
                errors.append(
                    f"F_mia: ターゲット列が不足（自チーム以外で{len(others)}列欠落: {others}）。"
                    "自チーム以外の全標的に答えること"
                )
        elif len(missing) > 1:
            errors.append(
                f"F_mia: ターゲット列が不足（全{len(allowed)}標的中{len(missing)}欠落: {missing}）。"
                "自チーム以外の全標的に答えること"
            )
    if dist and "attacker_record_ids" in dist:
        expected = dist["attacker_record_ids"]
        got = set(pd.to_numeric(df["record_id"], errors="coerce").dropna().astype(int).tolist())
        if got != expected:
            errors.append(
                f"F_mia: record_id集合が配布コホートと不一致（過不足）: 不足{len(expected - got)}行/"
                f"余分{len(got - expected)}行"
            )
    _check_unit_interval(df, target_cols, "F_mia", errors)
    return ValidationResult(len(errors) == 0, errors)


def validate_aia_submission(df: pd.DataFrame, cfg: dict, dist: dict | None = None) -> ValidationResult:
    """AIAロング [target_k, challenge_row_id, time_hat]。

    `onset_conf` は提出しない。採点は発症時刻の復元（±tau年）だけを使い、発症の有無は
    使わないため。採点に反映されない列を出させると参加者を混乱させる。
    """
    errors: list[str] = []
    cols_ok = submission_schema._check_columns(df, AIA_COLUMNS, "F_aia", errors)
    if not cols_ok:
        return ValidationResult(False, errors)

    th = pd.to_numeric(df["time_hat"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(th).any():
        errors.append("F_aia[time_hat]: 数値化できない/欠測がある")
    if (np.isfinite(th) & (th < 0.0)).any():
        errors.append("F_aia[time_hat]: 負値がある")

    tk = pd.to_numeric(df["target_k"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(tk).any() or not np.allclose(tk[np.isfinite(tk)], np.round(tk[np.isfinite(tk)])):
        errors.append("F_aia[target_k]: 整数でない/欠測がある")
    seen_targets = {int(x) for x in tk[np.isfinite(tk)]}
    # ★同じ (target_k, challenge_row_id) の重複を落とす。ルールブック §5.2＝「全行に1行ずつ」。
    #   下の過不足は集合で比べるので重複をすり抜け、採点（aia_truth との inner merge）では二重に数えられる。
    pair = pd.DataFrame({"k": tk, "r": pd.to_numeric(df["challenge_row_id"], errors="coerce")}).dropna()
    n_dup = int(pair.duplicated().sum())
    if n_dup:
        errors.append(f"F_aia: 同じ (target_k, challenge_row_id) の行が重複している（余分{n_dup}行）。"
                      "各行に1行ずつ答えること")
    if dist and "mia_columns" in dist:
        allowed = {int(x) for x in dist["mia_columns"]["target_columns"]}
        unknown = seen_targets - allowed
        if unknown:
            errors.append(f"F_aia: 未知のtarget_k: {sorted(unknown)}")
    if dist and "challenge" in dist:
        rid = pd.to_numeric(df["challenge_row_id"], errors="coerce")
        got: dict[int, set[int]] = {}
        for k, r in zip(tk, rid):
            if np.isfinite(k) and pd.notna(r):
                got.setdefault(int(k), set()).add(int(r))
        own = dist.get("team_id")
        for k, ids in dist["challenge"].items():
            # 自チームを対象にした行は採点で除くので、入っていても中身の過不足は問わない（§5.2）。
            if own is not None and int(k) == int(own):
                continue
            if k in seen_targets and got.get(k, set()) != ids:
                errors.append(f"F_aia: target_k={k} のchallenge_row_id集合が配布と不一致（過不足）")
        # ★標的そのものの欠落（F_miaと同じ理由）。自チーム分の1標的は欠けてよい。
        missing_k = sorted(set(dist["challenge"]) - seen_targets)
        if own is not None:
            # ★本番は自チーム以外の欠落を1つでも落とす（F_mia と同じ理由）。
            others_k = [k for k in missing_k if int(k) != int(own)]
            if others_k:
                errors.append(
                    f"F_aia: 標的が不足（自チーム以外で{len(others_k)}標的欠落: {others_k}）。"
                    "自チーム以外の全標的に答えること"
                )
        elif len(missing_k) > 1:
            errors.append(
                f"F_aia: 標的が不足（全{len(dist['challenge'])}標的中{len(missing_k)}欠落: {missing_k}）。"
                "自チーム以外の全標的に答えること"
            )
    return ValidationResult(len(errors) == 0, errors)


# --------------------------------------------------------------------------- #
# オーケストレータ ＋ dist ローダ
# --------------------------------------------------------------------------- #
def is_production_dist(dist_dir: str | Path | None) -> bool:
    """`dist_dir` が本番の真値ディレクトリか（`tokens.csv` があるか）。

    `scoring_io.is_production_reference` と同じ目印。検証器は採点アダプタ（codabench）に
    依存しない層なので、同じ判定をここにも置く。
    """
    return dist_dir is not None and (Path(dist_dir) / "tokens.csv").is_file()


def lookup_team_id(dist_dir: str | Path | None, token: str | None) -> int | None:
    """本番の真値ディレクトリの `tokens.csv` で token から team_id を引く。練習・未知トークンは None。

    `scoring_io.resolve_team_id` と同じ引き方（前後の空白を落として完全一致・重複時は後の行）。
    """
    if token is None or not is_production_dist(dist_dir):
        return None
    df = pd.read_csv(Path(dist_dir) / "tokens.csv", dtype={"token": str})
    hit = pd.to_numeric(df.loc[df["token"].astype(str).str.strip() == str(token).strip(), "team_id"],
                        errors="coerce").dropna()
    return int(hit.iloc[-1]) if len(hit) else None


def load_dist(dist_dir: str | Path, team_id: int | None = None) -> dict:
    """配布物から検証に使う参照を読む: mia_columns（ターゲット集合）とAIAチャレンジ集合。

    渡されたディレクトリを読むだけで、置き場所には依存しない。

    本番の真値ディレクトリはチーム番号つきの名前（`B_{id}.csv`・`pool_ids_{id}.csv`・
    `aia_truth_{id}.csv`）で、練習用の固定名（`B_practice.csv`・`pool_ids_attacker.csv`・
    `aia_challenge_*.csv`）を持たない。`team_id`（提出者。`lookup_team_id` で引く）を渡すと、
    C の行数（|C|==|B_i|）と F_mia の行キーを提出者のチーム番号つきの名前で読む。
    チャレンジ集合は、`aia_challenge_*.csv` が無ければ `aia_truth_*.csv` の `challenge_row_id` から作る
    （配布する `aia_challenge_<k>.csv` と同じ集合）。
    """
    dist_dir = Path(dist_dir)
    # 攻撃側参照の置き場: attacker/（事務局）・participant_data/targets（参加者配布）・
    # reference_data 直下（練習）／truth_dir 直下（本番）の3系統を見る。ここを取り違えると mia_columns / challenge が
    # 読めず、ターゲット過不足や行キーの検査が黙って省略される。
    attacker = dist_dir
    for cand in (dist_dir / "attacker", dist_dir / "targets"):
        if (cand / "mia_columns.json").exists() or any(cand.glob("aia_challenge_*.csv")):
            attacker = cand
            break
    d: dict = {}
    if team_id is not None:
        d["team_id"] = int(team_id)
    # 防御C_iの期待行数＝配布B_iの行数（|C|==|B_i|強制）。本番は提出者の `B_{id}.csv`、
    # 練習の参照データ・参加者配布はルート直下の B_practice.csv。無ければ渡さない
    # （None→上限のみの後方互換。本番で無いときは `production_ref_errors` が落とす）。
    b_practice = dist_dir / "B_practice.csv"
    b_self = dist_dir / f"B_{int(team_id)}.csv" if team_id is not None else b_practice
    if b_self.exists():
        d["defense_rows"] = int(len(pd.read_csv(b_self)))
    # F_mia の行キー照合用＝攻撃者自身のコホートの record_id 集合。本番は提出者の
    # `pool_ids_{id}.csv`。練習の reference_data には pool_ids_attacker.csv があり、
    # 参加者配布では B_practice.csv が同じ集合。
    pool_cands = ((dist_dir / f"pool_ids_{int(team_id)}.csv",) if team_id is not None
                  else (dist_dir / "pool_ids_attacker.csv", b_practice))
    for cand in pool_cands:
        if not cand.exists():
            continue
        _df = pd.read_csv(cand)
        if "record_id" not in _df.columns:
            continue  # record_id を持たない配布物（テスト用のC形式など）は行キー照合の材料にしない
        d["attacker_record_ids"] = set(_df["record_id"].astype(int).tolist())
        break
    mc = attacker / "mia_columns.json"
    if mc.exists():
        d["mia_columns"] = json.loads(mc.read_text(encoding="utf-8"))
    challenge: dict[int, set[int]] = {}
    for f in sorted(attacker.glob("aia_challenge_*.csv")):
        k = int(f.stem.rsplit("_", 1)[-1])
        challenge[k] = set(pd.read_csv(f)["challenge_row_id"].astype(int).tolist())
    if not challenge and is_production_dist(dist_dir):
        # 本番の真値ディレクトリには配布用の aia_challenge が無い＝答え合わせ用の aia_truth から作る。
        # 対象は mia_columns.json の標的に限る（それ以外の aia_truth が置かれていても標的に数えない）。
        targets = ({int(t) for t in d["mia_columns"]["target_columns"]} if "mia_columns" in d else None)
        for f in sorted(dist_dir.glob("aia_truth_*.csv")):
            suffix = f.stem.rsplit("_", 1)[-1]
            if not suffix.isdigit() or (targets is not None and int(suffix) not in targets):
                continue
            challenge[int(suffix)] = set(
                pd.read_csv(f, usecols=["challenge_row_id"])["challenge_row_id"].astype(int).tolist())
    if challenge:
        d["challenge"] = challenge
    # 希少検出の公開参照（schema.json 同梱）＝提出前の自己確認(機能#2)用。participant_data 直下か
    # defender/ 下も探す。無ければ黙って省略（旧配布物との後方互換）。
    for cand in (dist_dir / "schema.json", dist_dir / "defender" / "schema.json"):
        if cand.exists():
            try:
                sj = json.loads(cand.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                break
            if isinstance(sj, dict) and isinstance(sj.get("rare_detector"), dict):
                d["rare_detector"] = sj["rare_detector"]
            break
    return d


def production_ref_errors(kind: str | None, dist_dir: str | Path | None, dist: dict | None) -> list[str]:
    """本番で、提出者の検証に要る参照が揃っていなければ理由を返す（揃っていれば空）。

    本番（`tokens.csv` がある真値ディレクトリ）で参照が見つからないときは、検査を黙って省かずに
    **検証で落とす**（Failed＝提出回数を消費しない・ルールブック §5.5）。黙って省くと、
    行数や行キーの検査が効いていないことに気づけないため。
    提出者が解決できない（未知トークン）ときは `check_token_registered` が先に落とすので何もしない。
    練習・参加者の手元（`tokens.csv` が無い）も何もしない。
    """
    if not is_production_dist(dist_dir) or not dist or "team_id" not in dist:
        return []
    k = dist["team_id"]
    missing: list[str] = []
    if kind == "defense":
        if "defense_rows" not in dist:
            missing.append(f"B_{k}.csv")
    elif kind == "attack":
        if "attacker_record_ids" not in dist:
            missing.append(f"pool_ids_{k}.csv")
        if "mia_columns" not in dist:
            missing.append("mia_columns.json")
        else:
            have = set((dist.get("challenge") or {}).keys())
            lack = sorted(int(t) for t in dist["mia_columns"]["target_columns"] if int(t) not in have)
            missing += [f"aia_truth_{t}.csv" for t in lack]
    if not missing:
        return []
    return [f"採点側の参照データが見つからないため検証できません（{', '.join(missing)}）。"
            "提出物の問題ではありません。事務局へ連絡してください"]


def rare_count_diagnostic(c_df: pd.DataFrame, dist: dict | None) -> dict | None:
    """提出C で希少と検出される件数を返す（提出前の自己確認・機能#2）。dist に rare_detector が
    無ければ None。U_rare のゲート(§6.1.3)＝この件数が閾値未満だと希少の忠実度が 0 になり U_rare=0。
    採点器と同じ μ,Σ,band を使うので、CodaBench 採点と同じ件数になる。"""
    rd = (dist or {}).get("rare_detector")
    if not rd:
        return None
    mu = np.asarray(rd["mu"], dtype=float)
    sigma = np.asarray(rd["sigma"], dtype=float)
    cols = rd.get("columns") or list(schema_mod.CONTINUOUS)
    m = outlier_mod.mahalanobis(c_df, mu, sigma, cols)
    n = int((m >= float(rd["band"]["lo"])).sum())
    gate = int(rd.get("gate", 8))
    return {"rare_detected": n, "gate": gate, "below_gate": n < gate}


def validate_submission_zip(
    zip_path: str | Path, cfg: dict, dist_dir: str | Path | None = None, kind_override: str | None = None
) -> ValidationResult:
    """層1→層2を通して提出zipを検証する。層1で落ちたら層2は走らせない。"""
    pkg = check_package(zip_path, cfg)
    if not pkg.ok:
        return ValidationResult(False, list(pkg.errors))

    kind = kind_override or pkg.kind
    team_id = lookup_team_id(dist_dir, pkg.token) if dist_dir else None
    dist = load_dist(dist_dir, team_id) if dist_dir else None
    errors = list(pkg.errors) + check_token_registered(pkg.token, dist_dir) \
        + production_ref_errors(kind, dist_dir, dist)
    if kind == "defense":
        n_exp = dist.get("defense_rows") if dist else None
        errors += submission_schema.validate_c_submission(pkg.frames["C.csv"], cfg, n_expected=n_exp).errors
    elif kind == "attack":
        errors += validate_mia_submission(pkg.frames["F_mia.csv"], cfg, dist).errors
        errors += validate_aia_submission(pkg.frames["F_aia.csv"], cfg, dist).errors
    else:
        errors.append(f"提出種が不明: {kind}")
    return ValidationResult(len(errors) == 0, errors)


# --------------------------------------------------------------------------- #
# ディレクトリ入力版（CodaBenchが展開済の $input/res を直接検証する）
# --------------------------------------------------------------------------- #
def check_package_dir(res_dir: str | Path, cfg: dict) -> PackageResult:
    """展開済の提出ディレクトリ（CodaBench `$input/res`）をパッケージ検証する。

    zip版 `check_package` と同一規約（直下フラット・必須ファイル過不足・CSV parse・token形式）を、
    ディレクトリ表現に対して適用する。zipサイズ上限はここでは検査しない（CodaBenchが展開後に渡すため）。
    純関数（読み取りのみ）。
    """
    errors: list[str] = []
    rd = Path(res_dir)
    token_pat = cfg["submission"]["token_pattern"]
    if not rd.is_dir():
        return PackageResult(False, [f"提出ディレクトリが無い: {rd}"])

    flat_names: list[str] = []
    for p in sorted(rd.iterdir()):
        if p.name in _CODABENCH_INJECTED:
            continue  # CodaBench注入の非提出ファイル（metadata等）は無視
        if p.is_dir():
            errors.append(f"サブフォルダ混入（直下フラットでない）: {p.name}/")
        elif p.name.startswith("."):
            errors.append(f"隠しファイル混入: {p.name}")
        else:
            flat_names.append(p.name)

    names = set(flat_names)
    kind = _detect_kind(names)
    if kind is None:
        errors.append(f"提出種を判定できない（C.csv / F_mia.csv+F_aia.csv のいずれも直下に無い）: {sorted(names)}")
        return PackageResult(False, errors)

    required = DEFENSE_FILES if kind == "defense" else ATTACK_FILES
    missing, extra = required - names, names - required
    if missing:
        errors.append(f"必須ファイル不足: {sorted(missing)}")
    if extra:
        errors.append(f"余計なファイル: {sorted(extra)}")

    frames: dict[str, pd.DataFrame] = {}
    for name in sorted(names & required):
        if not name.endswith(".csv"):
            continue
        raw = (rd / name).read_bytes()
        if len(raw) == 0:
            errors.append(f"{name}: 空ファイル")
            continue
        try:
            df = submission_schema.read_scoring_csv(io.BytesIO(raw), encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{name}: UTF-8でデコードできない")
            continue
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: CSVとしてparseできない（{type(e).__name__}）")
            continue
        if df.shape[0] == 0:
            errors.append(f"{name}: データ行が無い")
        frames[name] = df

    token = None
    if "token.txt" in names:
        try:
            raw_txt = (rd / "token.txt").read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append("token.txt: UTF-8でデコードできない")
        else:
            lines = [ln for ln in raw_txt.splitlines() if ln.strip()]
            if len(lines) != 1:
                errors.append(f"token.txt: 1行でない（非空{len(lines)}行）")
            else:
                token = lines[0].strip()
                if not re.match(token_pat, token):
                    errors.append(f"token.txt: 形式不一致（{token_pat}）")

    return PackageResult(len(errors) == 0, errors, kind, frames, token)


def check_token_registered(token: str | None, dist_dir: str | Path | None) -> list[str]:
    """`token.txt` が登録済みトークンかを検査する（本番の reference_data に対してのみ効く）。

    `dist_dir` に `tokens.csv`（列 `token,team_id`）があれば本番参照データ＝一致を要求する。
    無ければ練習・参加者手元なので**何も検査しない**（参加者は tokens.csv を持たない）。

    未知のトークンでの提出は**提出回数を消費しません**。CodaBench 側で "Failed" に
    する必要があるため、採点ではなく**検証で落とします**（`run._run` が非ゼロ終了します）。
    ルールブック §5.5 と同じ扱いです。★正しいトークンが何かは示しません
    （総当たりの手掛かりにしないため）。
    """
    if dist_dir is None or token is None:
        return []
    p = Path(dist_dir) / "tokens.csv"
    if not p.is_file():
        return []
    known = set(pd.read_csv(p, dtype={"token": str})["token"].astype(str).str.strip())
    if str(token).strip() in known:
        return []
    return ["token.txt: 登録されているトークンと一致しません"
            "（配布物に同梱された token.txt をそのまま入れてください）"]


def _run_layer2(kind: str, frames: dict[str, pd.DataFrame], cfg: dict, dist: dict | None) -> list[str]:
    """層2（提出種別の論理検証）。zip版・dir版で共有。"""
    if kind == "defense":
        n_exp = dist.get("defense_rows") if dist else None
        return list(submission_schema.validate_c_submission(frames["C.csv"], cfg, n_expected=n_exp).errors)
    if kind == "attack":
        errs = list(validate_mia_submission(frames["F_mia.csv"], cfg, dist).errors)
        errs += validate_aia_submission(frames["F_aia.csv"], cfg, dist).errors
        return errs
    return [f"提出種が不明: {kind}"]


def validate_submission_dir(
    res_dir: str | Path, cfg: dict, dist_dir: str | Path | None = None, kind_override: str | None = None
) -> tuple[ValidationResult, PackageResult]:
    """展開済提出ディレクトリを層1→層2で検証し、(結果, パッケージ) を返す。層1で落ちたら層2は走らせない。

    CodaBench採点アダプタ用（frames を採点へ渡すため PackageResult も返す）。
    """
    pkg = check_package_dir(res_dir, cfg)
    if not pkg.ok:
        return ValidationResult(False, list(pkg.errors)), pkg
    kind = kind_override or pkg.kind
    # ★本番は提出者（token→team_id）を先に決め、その参照で行数・行キーを見る（`load_dist`）。
    team_id = lookup_team_id(dist_dir, pkg.token) if dist_dir else None
    dist = load_dist(dist_dir, team_id) if dist_dir else None
    errors = list(pkg.errors) + check_token_registered(pkg.token, dist_dir) \
        + production_ref_errors(kind, dist_dir, dist) + _run_layer2(kind, pkg.frames, cfg, dist)
    return ValidationResult(len(errors) == 0, errors), pkg


def main() -> None:
    parser = argparse.ArgumentParser(description="提出物フォーマット検証器（2層・starter kit自己チェック）")
    parser.add_argument("zip", help="提出zipのパス")
    parser.add_argument(
        "--config", default=None,
        help="採点・検証パラメータ（既定＝パッケージ同梱の kit_config.yaml）。"
             "別のファイルを指定することもできる。",
    )
    parser.add_argument("--kind", default="auto", choices=["auto", "defense", "attack"])
    parser.add_argument("--dist", default=None, help="配布物ディレクトリ（列/チャレンジ照合に使用）")
    args = parser.parse_args()

    cfg = config_mod.load_config(args.config or KIT_CONFIG_PATH)
    kind = None if args.kind == "auto" else args.kind
    res = validate_submission_zip(args.zip, cfg, args.dist, kind)
    # ★機能#2: 防御提出は「希少検出件数」を提出前に表示（U_rareゲートの自己確認・ルールブック §6.1.3）
    if args.dist:
        pkg = check_package(args.zip, cfg)
        if pkg.ok and (kind or pkg.kind) == "defense" and "C.csv" in pkg.frames:
            diag = rare_count_diagnostic(pkg.frames["C.csv"], load_dist(args.dist))
            if diag is not None:
                head = f"希少検出: C={diag['rare_detected']}件 / 閾値{diag['gate']}件"
                if diag["below_gate"]:
                    print(f"[注意] {head} → 希少群の忠実度を検証できず U_rare は 0 になります（ルールブック §6.1.3）。")
                else:
                    print(f"[情報] {head} → 希少件数はゲートを満たしています。")
    if res.ok:
        print("OK: 提出物は妥当です（層1パッケージ＋層2論理）")
        sys.exit(0)
    print("NG: 以下の問題があります:")
    for e in res.errors:
        print(f"  - {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
