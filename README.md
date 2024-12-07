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


## データセットについて(wip)
githubにデータは含まれていない

ここから落として使う: https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models

実際のdownload linkはこっち: https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/salt_and_overthrust_models.tar.gz