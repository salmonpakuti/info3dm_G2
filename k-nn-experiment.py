
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

<<<<<<< HEAD
max_score=0
=======
        file_path="ファイルパス"
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
>>>>>>> ce95bdb (recommit^^)


file_path="winequality-white-re.csv"
df = pd.read_csv(file_path,encoding="shift-jis")

#そのままの特徴量
#x=pd.DataFrame(df[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]])
#pca後の特徴量
#x=pd.DataFrame(df[["residual sugar","free sulfur dioxide","total sulfur dioxide","alcohol"]])
#umap後の特徴量
x=pd.DataFrame(df[["fixed acidity","volatile acidity","chlorides","total sulfur dioxide","density","alcohol"]])

y=pd.DataFrame(df[["quality"]])
#XとYを学習データとテストデータに分割
X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size=0.3, shuffle=True, random_state=3, stratify=y)
Y_train=np.reshape(Y_train,(-1))
Y_test=np.reshape(Y_test,(-1))

model = KNeighborsClassifier(
        #n_neighbors=65,
        weights="uniform",
        #algorithm="auto",
        #metric="minkowski"
        #p=1

        ) 


K_grid = {KNeighborsClassifier(): {"n_neighbors": [i for i in range(20,50)],
                           #"weights": ["uniform", "distance"],
                           "algorithm": ["auto","ball_tree","kd_tree","brute"],
                           "metric":["euclidean","manhattan","chebyshev","minkowski","hamming","canberra","braycurtis"],
                           "p": [1,2]}}

<<<<<<< HEAD
#グリッドサーチ
for model, param in K_grid.items():
    clf = GridSearchCV(model, param)
    clf.fit(X_train, Y_train)
    pred_y = clf.predict(X_test)
    score = f1_score(Y_test, pred_y, average="micro")
=======
        k_neighbors_learning(x,y)
        k_neighbors_gridsearch(x,y)
        val_curve(x,y,model)
>>>>>>> ce95bdb (recommit^^)

    if max_score < score:
        max_score = score
        best_param = clf.best_params_
        best_model = model.__class__.__name__

print("サーチ方法:グリッドサーチ")
print("ベストスコア:{}".format(max_score))
print("モデル:{}".format(best_model))
print("パラメーター:{}".format(best_param))