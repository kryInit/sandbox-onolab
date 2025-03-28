## how to use(wip)
```bash
poetry install
poetry run python src/{...}.py
```

## directory構成
|      path      | description                                                     |
|:--------------:|:----------------------------------------------------------------|
|     `lib`      | ライブラリとしてのソースコード                                                 |
|     `src`      | メインのソースコード(entry point)                                         |
|   `sandbox`    | 実験的ソースコード                                                       |
|   `notebook`   | ノートブック形式の実験的ソースコード                                              |
| `data-storage` | データセットの格納ディレクトリ<br/>`lib.misc` などで定義されているパスから読み取られるが、それ以上の意味はない |
|    `output`    | 出力データの格納ディレクトリ<br/>`lib.misc` などで定義されているパスから読み取られるが、それ以上の意味はない  |
|    `thesis`    | 論文の原稿                                                           |
|   `frontend`   | 結果表示用web pageのフロントエンドコード（updateされていない）                          |


## `lib.seismic.devito_example`について
使用するライブラリDevitoはpython libraryとしてPyPI上で提供されている: https://pypi.org/project/devito/

しかし、Devitoは汎用的な微分方程式をシミュレーションするライブラリであり、seismicに適用するためにはdevitoのリポジトリに存在するラッパーコードを使用すると便利に扱える: https://github.com/devitocodes/devito/tree/master/examples/seismic

そのため、`lib.seismic.devito_example` に、[ここ](https://github.com/devitocodes/devito/tree/2271407b98c9d4c4d9e049e6de91e5eb4e17285b/examples/seismic) からcloneしたコードを配置している

上記のラッパーコードを使用せずともシミュレーションは可能であり、例えば `sandbox.seismic.seismic_modeling{1,3}.py` などはdevito libraryのみを用いてシミュレーションを行なっている

ただし、その場合はdamping層の設定や入力データの設定を自前で行わなければならないため、ラッパーコードを用いた方が便利ではある（が、ライブラリとして提供されていない微妙に整理されていないコードをリポジトリに含めるのは気持ちが悪い, そもそも普通にscipyでやりたい気持ちはあるが、雑に実装するとdevitoの方が速かったのでこちらを使っている・・・


## データセットについて
githubにデータは含まれていない

ここから落として使う: 
- salt and overthrust models: https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models
- BP2004: https://wiki.seg.org/wiki/2004_BP_velocity_estimation_benchmark_model

- `src/SIP-box-TV-constrained-FWI.py`は`{project-root}/data-storage/salt-and-overthrust-models/3-D_Salt_Model/VEL_GRIDS/Saltf@@`を参照している
- `src/GRSL-box-TV-constrained-FWI.py`は`{project-root}/data-storage/BP2004/vel_z6.25m_x12.5m_exact.segy`を参照している
- （適宜変更してください）