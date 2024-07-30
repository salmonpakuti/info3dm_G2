import pytest
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

class knnTest:
    """_summary_
    このクラスは、k-nn-experimentのユニットテストを行うクラスです。
    """

    @classmethod
    def generate_test_data(cls):
        """テスト用のデータを生成するメソッド
        X:特徴量
        y:正解ラベル
        """
        np.random.seed(0)
        X = np.random.rand(100, 6)  # 100サンプル、6特徴量
        y = np.random.randint(0, 2, size=100)  # 0から1までの品質スコア
        return X, y

    @classmethod
    def test_k_neighbors_learning(cls):
        """
        k_neighbors_learning 関数のテストを行うメソッド。正解率が出力されることを確認する。
        """
        X, y = cls.generate_test_data()
        
        def k_neighbors_learning(a, b):
            X_train, X_test, Y_train, Y_test = train_test_split(a, b, test_size=0.3, shuffle=True, random_state=3, stratify=b)
            model = KNeighborsClassifier(n_neighbors=22, weights="uniform", algorithm="auto", metric="canberra")
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            accuracy = accuracy_score(Y_test, Y_pred)
            return accuracy
        
        accuracy = k_neighbors_learning(X, y)
        # 正解率が出力されることを確認
        print("正解率: ", accuracy)
        
    @classmethod
    def test_k_neighbors_gridsearch(cls):
        """
        k_neighbors_gridsearch 関数のテストを行うメソッド
        """
        X, y = cls.generate_test_data()
        
        def k_neighbors_gridsearch(a, b):
            
            #algorithm=brute
            K_grid = {KNeighborsClassifier(): {
                "n_neighbors": [i for i in range(20, 50)],
                "algorithm": ["brute"],
                "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "hamming", "canberra", "braycurtis"],
                "p": [1, 2]
            }}
            """
            algorithmごとに使えるmetricが異なるので、分けている

            #algorithm=auto
            K_grid = {KNeighborsClassifier(): {
                "n_neighbors": [i for i in range(20, 50)],
                "algorithm": ["auto"],
                "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "hamming", "canberra", "braycurtis"],
                "p": [1, 2]
            }}
            #algorithm=ball_tree
            K_grid = {KNeighborsClassifier(): {
                "n_neighbors": [i for i in range(20, 50)],
                "algorithm": ["ball_tree"],
                "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "hamming", "canberra", "braycurtis"],
                "p": [1, 2]
            }}
            #algorithm=kd_tree
            K_grid = {KNeighborsClassifier(): {
                "n_neighbors": [i for i in range(20, 50)],
                "algorithm": ["kd_tree"],
                "metric": ["manhattan", "chebyshev", "minkowski"],
                "p": [1, 2]
            }}
            
            """
            X_train, X_test, Y_train, Y_test = train_test_split(a, b, test_size=0.3, shuffle=True, random_state=3, stratify=b)
            
            max_score = 0
            best_param = None
            best_model = None
            
            for model, param in K_grid.items():
                clf = GridSearchCV(model, param)
                clf.fit(X_train, Y_train)
                pred_y = clf.predict(X_test)
                score = f1_score(Y_test, pred_y, average="micro")
                
                if max_score < score:
                    max_score = score
                    best_param = clf.best_params_
                    best_model = model.__class__.__name__

            return max_score, best_param, best_model
        
        max_score, best_param, best_model = k_neighbors_gridsearch(X, y)
        
        # 最大スコアが期待される範囲にあることを確認する
        assert max_score >= 0.0  # スコアの下限を0.0と仮定

    @classmethod
    def test_val_curve(cls):
        """
        val_curve 関数のテストを行うメソッド。（可視化のテストは難しいため、データが生成されるかどうかの確認を行う）
        """
        X, y = cls.generate_test_data()
        
        def val_curve(a, b, model1):
            param_range = [5, 10, 15, 20, 25, 30, 35, 40]
            train_scores, test_scores = validation_curve(
                estimator=model1,
                X=a, y=b,
                param_name="n_neighbors",
                param_range=param_range, cv=10)
            
            return train_scores, test_scores
        
        model = KNeighborsClassifier()
        train_scores, test_scores = val_curve(X, y, model)
        # 訓練スコアとテストスコアの形状を確認する
        assert train_scores.shape == (8, 10), f"Expected shape (8, 10), but got {train_scores.shape}"
        assert test_scores.shape == (8, 10), f"Expected shape (8, 10), but got {test_scores.shape}"

if __name__ == "__main__":
    #print("k_neighbors_learningのユニットテスト")
    #knnTest.test_k_neighbors_learning()
    #print("k_neighbors_gridsearchのユニットテスト")
    #knnTest.test_k_neighbors_gridsearch()
    print("val_curveのユニットテスト")
    #knnTest.test_val_curve()

