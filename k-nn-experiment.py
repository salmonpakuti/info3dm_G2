import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class knn:
        """_summary_
        このクラスは、k近傍法でワインの特徴量を学習、グリッドサーチでパラメータ調整、検証曲線の表示を行うクラスです。
        filepath(str):データセットのファイルパス
        df(dataframe):csvファイルを読み込み、データフレーム型として格納します
        x(dataframe):特徴量
        y(dataframe):正解ラベル
        model:学習で利用するモデル
        """
        
        

        file_path="/Users/yuna/winequality-white-re.csv"
        df = pd.read_csv(file_path,encoding="shift-jis")
        #そのままの特徴量
        #x=pd.DataFrame(df[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]])
        #pca後の特徴量
        #x=pd.DataFrame(df[["residual sugar","free sulfur dioxide","total sulfur dioxide","alcohol"]])
        #umap後の特徴量
        x=pd.DataFrame(df[["fixed acidity","volatile acidity","chlorides","total sulfur dioxide","density","alcohol"]])
        y=pd.DataFrame(df[["quality"]])
        model=KNeighborsClassifier()

        def k_neighbors_learning(a,b):
                """_summary_：k近傍法でワインの品質を学習を行い、精度を出力するメソッド
                Args:
                a (dataframe): 特徴量
                b (dataframe): 正解ラベル
                """
                #XとYを学習データとテストデータに分割
                X_train,X_test,Y_train,Y_test = train_test_split(a,b, test_size=0.3, shuffle=True, random_state=3, stratify=b)
                Y_train=np.reshape(Y_train,(-1))
                Y_test=np.reshape(Y_test,(-1))

                model = KNeighborsClassifier(
                        n_neighbors=22,
                        weights="uniform",
                        algorithm="auto",
                        metric="canberra"
                        #p=1
                        ) 
                model.fit(X_train,Y_train)
                Y_pred_tree=model.predict(X_test)
                print(f'正解率: {accuracy_score(Y_test, Y_pred_tree)}')



        def k_neighbors_gridsearch(a,b):
                """_summary_：k近傍のグリッドサーチを行い、最適なパラメータを出力するメソッド

                Args:
                    a (dataframe): 特徴量
                    b (dataframe): 正解ラベル
                """
                max_score=0
                #XとYを学習データとテストデータに分割
                X_train,X_test,Y_train,Y_test = train_test_split(a,b, test_size=0.3, shuffle=True, random_state=3, stratify=b)
                Y_train=np.reshape(Y_train,(-1))
                Y_test=np.reshape(Y_test,(-1))

                K_grid = {KNeighborsClassifier(): {"n_neighbors": [i for i in range(20,50)],
                                        #"weights": ["uniform", "distance"],
                                        "algorithm": ["auto","ball_tree","kd_tree","brute"],
                                        "metric":["euclidean","manhattan","chebyshev","minkowski","hamming","canberra","braycurtis"],
                                        "p": [1,2]}}

                #グリッドサーチ
                for model, param in K_grid.items():
                        clf = GridSearchCV(model, param)
                        clf.fit(X_train, Y_train)
                        pred_y = clf.predict(X_test)
                        score = f1_score(Y_test, pred_y, average="micro")

                if max_score < score:
                        max_score = score
                        best_param = clf.best_params_
                        best_model = model.__class__.__name__

                print("サーチ方法:グリッドサーチ")
                print("ベストスコア:{}".format(max_score))
                print("モデル:{}".format(best_model))
                print("パラメーター:{}".format(best_param))

        #検証曲線
        def val_curve(a,b,model1):
                """_summary_：k近傍のパラメータに関して検証曲線を描写するメソッド

                Args:
                    a (dataframe): 特徴量
                    b (dataframe): 正解ラベル
                    model1 (sklearn.neighbors._classification.KNeighborsClassifier): 検証曲線を描写したいモデル
                """
                #param_range=[4,8,12,16,20,24]
                param_range=[5,10,15,20,25,30,35,40]
                #param_range=[1,2]
                train_scores, test_scores = validation_curve(
                        estimator=model1,
                        X=a, y=np.reshape(b,(-1)),
                        param_name="n_neighbors",
                        param_range=param_range, cv=10)

                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)
                plt.figure(figsize=(8, 6))
                plt.plot(param_range, train_mean, marker='o', label='Train accuracy')
                plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.2)
                plt.plot(param_range, test_mean, marker='s', linestyle='--', label='Validation accuracy')
                plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.2)
                plt.grid()
                plt.xscale('log')
                plt.title('Validation curve (wminkowski)', fontsize=16)
                plt.xlabel('n_neighbors', fontsize=12)
                plt.ylabel('Accuracy', fontsize=12)
                plt.legend(fontsize=12)
                plt.ylim([0.6, 1.05])
                plt.show()



        #k_neighbors_learning(x,y)
        #k_neighbors_gridsearch(x,y)
        #val_curve(x,y,model)


