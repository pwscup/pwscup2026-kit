# PWS Cup 2026 スターターキット

匿名化技術コンテスト **PWS Cup 2026** の参加者向け配布物一式です。**練習用のデータ**（本番とは別に作った「トイ世界」）で、提出から採点までひととおり試せます。

競技のルールと採点の定義は **`rulebook/` の PDF が正本**です。このリポジトリのコードや説明と食い違ったら、ルールブックが正しいと考えてください。

- 公式ページ: <https://www.iwsec.org/pws/2026/cup26.html>
- 参加登録: 公式ページの参加申込フォームから（2026/07/30 〜 2026/09/11 受付。8/25 以降のお申し込みは本戦・加工フェーズ 9/12 からの参加になります）
- 提出先: [CodaBench](https://www.codabench.org/competitions/17698)
- 問い合わせ: pwscup2026-info (at) csec.ipsj.or.jp

---

## 中身

| パス | 何か |
|---|---|
| `rulebook/PWS_Cup2026_ルールブック_main-process-20260912.pdf` | **競技ルールの正本**。得点の定義・提出様式・禁止事項 |
| `participant_data/` | **練習用データ一式**。下の「配布データ」参照 |
| `notebooks/` | **教材（任意）**。ノートブック7本（「はじめてのPWSCup」＝提出まで1周／「採点のしくみ」＝点数の中身／有用性の指標を1つずつ確かめる5本）と、**読みもの1本**（コードなし） |
| `starter/` | **コード**。提出物の検証器・ローカル自己採点・参照攻撃・参照防御 |
| Docker イメージ | `hajimeono/pwscup2026-kit:main-process-20260912`（Docker Hub・public・amd64/arm64）。**Python 環境なしで検証と自己採点ができます** |
| `LICENSE` | コードのライセンス（Apache-2.0） |

**リポジトリごとダウンロードする場合**は、[Releases](https://github.com/pwscup/pwscup2026-kit/releases) の先頭（`Latest` が付いているもの）から `Source code (zip)` を落としてください。Releases はフェーズごとに中身を固定してあるので、**あとから同じ状態を取り直せます**。そのフェーズのルールブック PDF も一緒に付いています。

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

**読むだけなら GitHub 上でそのまま表示されます**——[`notebooks/はじめてのPWSCup.ipynb`](notebooks/はじめてのPWSCup.ipynb) / [`notebooks/採点のしくみ.ipynb`](notebooks/採点のしくみ.ipynb)。下の読みものは markdown なので、動かす準備も要りません。

| ノートブック | 中身 | 所要 |
|---|---|---|
| はじめてのPWSCup | データの読み方 → 素朴な匿名化 → 自己採点 → 素朴な攻撃 → 提出物の書き出し | 20〜30 分 |
| 採点のしくみ | 4観点が何を測っているか・**どの操作で点が上下するか**を、加工を作りながら確かめる | 15〜20 分 |
| 有用性の指標（5本） | 打ち切り整合・相関の一致・裾の一致・重複行・行の新規性。ルールブックの式どおりに計算して、自己採点の値と一致することを確かめる | 各 5〜15 分 |
| [読みもの] 競技データはどう作られているか | 合成データの材料と作り方のうち、開示する範囲（コードなし） | 5 分 |

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
docker run --rm -v "$PWD":/w hajimeono/pwscup2026-kit:main-process-20260912 \
  validate /w/my_defense.zip --dist /w/participant_data

# 有用性の自己採点（U と4観点の内訳）
docker run --rm -v "$PWD":/w hajimeono/pwscup2026-kit:main-process-20260912 \
  score /w/C.csv --dist /w/participant_data
```

**★ タグ `main-process-20260912` を省略しないでください。** `:latest` はフェーズが進むと別のイメージを指すようになります。タグを書いておけば、あとで同じ点数を再現できます。

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
| `reference/run_defense.py` | **参照防御**。既定の `copula_survival` は、`B` から、列ごとの値の分布と、年齢や検査値どうしの関係（相関）を学んで、同じ人数の架空の人を作り直します（コピュラという手法）。転帰（`onset`・`death`・`time`）は、`B` に当てはめた Cox モデル（年齢・性別・喫煙・検査値など〔共変量〕から、発症や死亡の起こりやすさを表すモデル）を使って、作った人の共変量に応じて引きます。`--method copula_synth` を付けると、転帰を共変量と無関係に引く素朴版になります |
| `reference/run_mia.py` | **参照MIA**。候補の人の近くに、`C` の人がどれだけ密に集まっているかと、`A_bg` の人がどれだけ密に集まっているかを、年齢と検査値の列で推定し（カーネル密度推定＝KDE）、`C` 側の密度が `A_bg` 側に比べて高い人ほど「`C` の元データ `B` にいた」と答えます（ルールブック §4.1） |
| `reference/run_aia.py` | **参照AIA**。challenge の各行について、性別と県が同じ `C` の行の中から、年齢・SBP・TG・HDL・ALT の値がいちばん近い行を1件探し（最近傍）、その行の `time` を答えます（ルールブック §4.2） |
| `reference/strong_synth.py` | **強い合成器**（ARF・CTGAN。どちらも、表全体を機械学習のモデルに学ばせて架空の行を作る、汎用の合成器）。追加依存が要ります。下記参照 |

出来た `C.csv` は `token.txt` と一緒に zip 直下に入れて提出します。

```
my_defense.zip
├── C.csv
└── token.txt
```

**★ 提出したら、リーダーボードに載せてください。** CodaBench は提出しただけでは
リーダーボードに載りません。提出一覧（My Submissions）の Actions 列のいちばん左のアイコン
（マウスを重ねると **Add to Leaderboard** と出ます）を押します。載っている提出はその列が
緑のチェックに変わります。**加工・攻撃のどちらのフェーズでも、順位の計算に使われるのは、
そのフェーズ終了時点でリーダーボードに載せている1件だけ**です。1件も載せずに終わった場合は
事務局が最新の有効な提出をあとから掲載しますが（ルールブック §5.3）、「最新」が自分の
いちばん良い提出とは限りません。**出したいものは自分で載せてください。**

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

データの生成方法（世界生成のコード）は競技の公平性のため非公開です。作り方のうち開示する範囲は [`notebooks/データはどう作られているか.md`](notebooks/データはどう作られているか.md) にまとめています。

© 2026 PWS Cup 2026 事務局（情報処理学会 コンピュータセキュリティ研究会 PWS組織委員会）
