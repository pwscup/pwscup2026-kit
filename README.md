# PWS Cup 2026 スターターキット

匿名化技術コンテスト **PWS Cup 2026** の参加者向け配布物一式です。**練習用のデータ**（本番とは別に作った「トイ世界」）で、提出から採点までひととおり試せます。

競技のルールと採点の定義は **`rulebook/` の PDF が正本**です。このリポジトリのコードや説明と食い違ったら、ルールブックが正しいと考えてください。

- 公式ページ: <https://www.iwsec.org/pws/2026/cup26.html>
- 参加登録: 公式ページの参加申込フォームから（説明会 7/30 から予備戦終了まで受付）
- 提出先: CodaBench（コンペページのURLは公式ページで案内します）
- 問い合わせ: pwscup2026-info (at) csec.ipsj.or.jp

---

## 中身

| パス | 何か |
|---|---|
| `rulebook/PWS_Cup2026_ルールブック_orientation-20260730c.pdf` | **競技ルールの正本**。得点の定義・提出様式・禁止事項 |
| `participant_data/` | **練習用データ一式**。下の「配布データ」参照 |
| `notebooks/` | **教材（任意）**。「はじめてのPWSCup」＝提出まで1周／「採点のしくみ」＝点数の中身 |
| `starter/` | **コード**。提出物の検証器・ローカル自己採点・参照攻撃・参照防御 |
| Docker イメージ | `hajimeono/pwscup2026-kit:slim-20260728b`（Docker Hub・public・amd64/arm64）。**Python 環境なしで検証と自己採点ができます** |
| `LICENSE` | コードのライセンス（Apache-2.0） |

**リポジトリごとダウンロードする場合**は、GitHub の緑の「Code」→「Download ZIP」が手軽です。

### 配布データ（`participant_data/`）

| ファイル | 何か |
|---|---|
| `B_practice.csv` | あなたのコホート（加工対象）。1,049 行 × 14 列 |
| `B_self.csv` | 自己採点用の拡張版 B（真の希少ラベル `is_rare` 付き）。**提出物ではありません** |
| `A_bg.csv` | 背景データ。攻撃の参照分布に使います |
| `targets/C_<k>.csv` | 攻撃対象の匿名化データ（練習では k = 1〜29 を公開） |
| `targets/aia_challenge_<k>.csv` | 属性推論の標的ファイル（1枚 326 行） |
| `targets/mia_columns.json` | その回の対象チームID一覧 |
| `sample_submissions/` | **そのまま提出できる形の実例**（`C.csv` / `F_mia.csv` / `F_aia.csv` / `token.txt`） |
| `schema.json` | 列と値域の定義 |
| `utility_ref.json` | 自己採点が使う参照統計 |
| `scoring_config.yaml` | 採点・検証パラメータ |

**まず `participant_data/sample_submissions/` の中身を開いて、出力の形をそのまま真似るのが一番早いです。**

提出物は CSV とテキストファイルだけなので、**Python 以外の言語・ツール（R / Julia / Excel / 手作業）で作ってもかまいません。** ただし、下の自己採点器・検証器を動かすには Python か Docker が要ります。

---

## いちばん短い道: ノートブック

**読むだけなら GitHub 上でそのまま表示されます**——[`notebooks/はじめてのPWSCup.ipynb`](notebooks/はじめてのPWSCup.ipynb) / [`notebooks/採点のしくみ.ipynb`](notebooks/採点のしくみ.ipynb)。

| ノートブック | 中身 | 所要 |
|---|---|---|
| はじめてのPWSCup | データの読み方 → 素朴な匿名化 → 自己採点 → 素朴な攻撃 → 提出物の書き出し | 20〜30 分 |
| 採点のしくみ | 4観点が何を測っているか・**どの操作で点が上下するか**を、加工を作りながら確かめる | 15〜20 分 |

実行環境は `starter/` に一本化しています（採点器と同じ版のライブラリで動くので、ノートブックの中で出る点数がそのままサーバの点数になります）。

```sh
cd starter
uv sync --extra notebook
uv run jupyter lab ..
```

---

## 提出物を検証する / 自己採点する

**有用性 U は CodaBench に出さなくても手元で採点できます**（同じ点数が出ます）。加工の作り込みはこのループが速いです。

### Docker（Python 環境が要りません・いちばん手軽）

```sh
# 提出 zip の形式チェック
docker run --rm -v "$PWD":/w hajimeono/pwscup2026-kit:slim-20260728b \
  validate /w/my_defense.zip --dist /w/participant_data

# 有用性の自己採点（U と4観点の内訳）
docker run --rm -v "$PWD":/w hajimeono/pwscup2026-kit:slim-20260728b \
  score /w/C.csv --dist /w/participant_data
```

**★ タグ `slim-20260728b` を省略しないでください。** `:latest` は古い版を指しており、`U_rare` がサーバ採点とわずかにずれます。

### Python（`starter/` を入れる）

uv を使う場合（推奨）。**同梱の `uv.lock` どおりの環境**が作られます。

```sh
cd starter
uv sync
uv run python -m pwscup2026.kit.validate ../my_defense.zip --dist ../participant_data
uv run python -m pwscup2026.kit.selfscore ../C.csv --dist ../participant_data
```

pip を使う場合:

```sh
cd starter
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
python -m pwscup2026.kit.selfscore ../C.csv --dist ../participant_data
```

**環境の版について。** 同梱の `uv.lock` は事務局が動作確認した環境です（この環境で参照防御の C を作り、自己採点まで通しています）。`starter/requirements-kit.txt` は**採点 Docker イメージそのものの版**なので、サーバ採点と厳密に合わせたい場合はこちらを使ってください（`uv pip install -r requirements-kit.txt`）。採点に効く numpy・pandas・scipy・scikit-learn・lifelines・statsmodels の版は両者で同一で、違うのは数値に関わらない推移的依存だけです。

**保護（protection）と攻撃力は他チームの情報が要るため、手元では採点できません。**

---

## 参照実装（実装例）

ルールブック §4.1・§4.2・§8 で「配ります」と書いている実装です。**方法を縛るものではありません**——独自の防御・攻撃を使ってかまいません。

```sh
cd starter
uv sync
uv run python reference/run_defense.py --dist ../participant_data --out ../C.csv
uv run python reference/run_mia.py     --dist ../participant_data --out ../F_mia.csv
uv run python reference/run_aia.py     --dist ../participant_data --out ../F_aia.csv
```

| スクリプト | 中身 |
|---|---|
| `reference/run_defense.py` | **参照防御**。`copula_survival`＝共変量をコピュラで再合成し、転帰は競合リスク Cox から共変量条件付きにサンプル（`--method copula_synth` で転帰独立の素朴版も選べます） |
| `reference/run_mia.py` | **参照MIA**。密度比 p_syn/p_ref を KDE で推定（ルールブック §4.1） |
| `reference/run_aia.py` | **参照AIA**。QI の最近傍を C から1件引いて `time` を答える（ルールブック §4.2） |
| `reference/strong_synth.py` | **強い合成器**（ARF・CTGAN）。追加依存が要ります。下記参照 |

出来た `C.csv` は `token.txt` と一緒に zip 直下に入れて提出します。

```
my_defense.zip
├── C.csv
└── token.txt
```

参照防御をそのまま回すと、有用性の4観点のうち **U_spec（特定の解析での結論一致）が最も低く出ます**。`--method copula_synth`（転帰を共変量と独立に引く素朴版）と見比べると、4観点が何を見ているかが分かります。**この比較は `notebooks/採点のしくみ.ipynb` で実際に採点しながら確かめられます。**

### 強い合成器（任意・追加依存）

```sh
cd starter
uv sync --extra strong-synth        # arfpy
uv run python reference/strong_synth.py --dist ../participant_data --method arf_synth --out ../C.csv

uv sync --extra ctgan               # sdv（torch を引きます・重いです）
uv run python reference/strong_synth.py --dist ../participant_data --method ctgan_synth --out ../C.csv
```

`copula_survival` は追加依存なしで使えます（`run_defense.py` の既定）。

---

## ライセンスと利用条件

- **コード**（`starter/` 以下）: Apache-2.0（`LICENSE`）
- **データとルールブック**: 参加・成果発表・研究・教育目的での利用を許諾します。第三者への再配布の条件は追ってご案内します。

データの生成方法（世界生成のコード）は競技の公平性のため非公開です。生成方法の開示範囲はルールブックに記載しています。

© 2026 PWS Cup 2026 事務局（情報処理学会 コンピュータセキュリティ研究会 PWS組織委員会）
