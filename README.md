# ワインの品質予測における多様な分類モデルの比較および検証

## 概要
私たちは．ワインの品質を二値で分類するために適切な分類モデルと次元削減手法の組み合わせの探索を行いました．分類手法は．SVC．K近傍法．Random Forestを用い．次元削減手法には．PCAとUMAPを用いました．今回はワイン品質予測において最も精度の良い分類手法を特定することを目的としましたが．これらのライブラリは他の分類問題での最適な手法を探索することもできると考えています．


## データセット(csv形式)
私たちは．kaggleで公開されているwine-quality-whiteを用いました．次元数は12個．データ数は4899個あります．

## 動作環境
Macで動作確認。
- python 3.9.6
- matplotlib 3.8.4
- numpy 1.26.4
- pandas 2.2.2
- pytest 8.3.1
- scikit-learn 1.4.2


## 使用方法
### PCA・UMAP
ipynb形式では，データセットをダウンロードし，コンパイルすると，寄与率のグラフ・表を出力してくれる.
また，py形式ファイルでは，プログラムの変数file_nameに代入しているパスを自身のデータセットのバスに変更し，コンパイルすると，寄与率のグラフ・表を出力してくれる.

### svc_experiment
svc_experiment.htmlはソースの詳細を記している．
まず，使用するデータセットのファイルパスをfilepathに格納する．次に，svc_learningのパラメータ．svc_gridsearchの探索範囲のパラメータ．plot_validation_curve関数内の検証曲線を描写したいパラメータとパラメータのとる値を設定する．最後にファイルを実行すると．精度．最適パラメータと最適パラメータでの精度．検証曲線が描写される．svc_experimentのユニットテストは．test_svcによって行なった．

### k-nn-experiment
k-nn-experiment.htmlはソースの詳細を記している．
まず．データセットをfilepathに格納する．次に．k_neighbors_learningのパラメータ．k_neighbors_gridsearchの探索範囲のパラメータ．val_curve関数内の検証曲線を描写したいパラメータとパラメータのとる値を設定する．最後にファイルを実行すると．精度．最適パラメータと最適パラメータでの精度．検証曲線が描写される．k-nn-experimentのユニットテストは．test_knnによって行なった．

### randomforest_experiment
まず，データセットをdata_pathに格納する．次に，RandomForestClassifierのパラメータ（n_estimators，max_depth，max_leaf_nodes=None，min_samples_split）を設定する．最後に，train_and_evaluate_rf_model(data_path)として実行すれば、精度が表示される．randomforest_experimentのユニットテストは，test_randamforest.pyによって行った。


## 作者
Contributors

