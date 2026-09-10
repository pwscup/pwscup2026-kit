PWS Cup 2026 練習キット — 参加者データ（トイ世界・全公開）

[配布物の一覧]
  記号        意味                                          誰が持つか    このキットでの配布
  --------------------------------------------------------------------------------------
  母集団      日本全国を模した仮想の人口全体                事務局のみ    ×
  A_bg        母集団のうち、どのチームのコホートにも入って  全員          ○ A_bg.csv
              いない人たちの表（背景の参照用）
  B_i         あなたのコホート。加工の入力                  自チームのみ  ○ B_practice.csv
  B_self      B_i に真値 is_rare を1列足した自己採点用      自チームのみ  ○ B_self.csv
  C_i         あなたが提出する匿名化済みの表                提出物        —（あなたが作る）
  C_k         他チーム k の C。攻撃の標的                    攻撃時に公開  ○ targets/C_<k>.csv
  対照        AIAの答え合わせで「見なくても当たる分」を測る  事務局のみ    ×
              ための、標的と条件をそろえた人たち
  challenge   AIAで推定する対象の行。標的の人と対照の人が    全員          ○ targets/aia_challenge_<k>.csv
              混ざっていて区別できない

  ※配布されるのは A_bg だけです（母集団そのものも、対照の名簿も配られません）。

[提出のしかた]
練習は1フェーズで、加工（防御）と攻撃のどちらでも提出できます。
防御練習: B_practice.csv を加工して C.csv を作り token.txt と一緒に提出。
攻撃練習: targets/ の C_<k>.csv・aia_challenge_<k>.csv を攻撃して F_mia.csv・F_aia.csv を作る。
提出フォーマットの詳細は CodaBench の Submission Format ページ、実例は sample_submissions/ を参照。
CodaBench の提出は zip のみ。そのまま提出できる例zip:
  sample_submissions/process_sub.zip（加工）・sample_submissions/attack_sub.zip（攻撃）

[ローカルでの自己採点]
B_self.csv は B_practice.csv に真値 is_rare 列を足した「自己採点用の拡張版B」です。
有用性はサーバに出さなくても手元で採点できます（CodaBenchと同じ点数。participant_data/ をカレントディレクトリにして実行）:
  docker run --rm -v "$PWD":/w hajimeono/pwscup2026-kit:main-process-20260912 score /w/C.csv --dist /w
utility_ref.json・scoring_config.yaml は採点器が使う参照値とパラメータです（消さないでください）。
※保護(protection)・攻撃力は他チームの情報が要るため手元では採点できません（サーバのみ）。

[ノートブック教材（任意）]
キットの notebooks/ に2本あります（このフォルダの外・スターターキットの直下）。
  はじめてのPWSCup.ipynb … 加工と攻撃を1周ずつ体験する（20〜30分）
  採点のしくみ.ipynb     … 得点がどう決まるか・何をすると点が上下するか（15〜20分）
  cd starter && uv sync --extra notebook && uv run jupyter lab ..
教材であってルールの正本ではありません。正本はルールブックと sample_submissions/ です。
提出物は CSV とテキストファイルだけなので、Python 以外の言語・ツールで作ってもかまいません
（自己採点器・検証器を動かすには Python か Docker が要ります）。

[ライセンス]
コード（提出フォーマット検証器・ローカル自己採点器・攻撃/加工の実装例・入門ノートブック）
  … Apache License 2.0（同梱の LICENSE を参照）
練習用データ（B_practice.csv・B_self.csv・A_bg.csv・targets/ など）およびルールブック
  … © 2026 PWS Cup 2026 事務局（情報処理学会 コンピュータセキュリティ研究会 PWS組織委員会）
     本コンテストへの参加、および参加チームによる成果発表・研究・教育目的での利用を許諾します。
     第三者への再配布や改変版の公開の条件は、追ってご案内します。

※本番（予備戦/本戦）のデータは別途配布されます（このキットは練習用）。
