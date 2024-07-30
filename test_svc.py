import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score

class SVCTest:
    """
    このクラスは、SVCのユニットテストを行うクラスです。
    """

    @classmethod
    def generate_test_data(cls):
        """
        テスト用のデータを生成するメソッド
        x:特徴量
        y:正解ラベル
        """
        np.random.seed(0)
        x = np.random.rand(100, 6)  # 100サンプル、6特徴量
        y = np.random.randint(0, 2, size=100)  # 0から1までの品質スコア
        return x, y

    @staticmethod
    def svc_learning(a, b):
        """
        SVCで学習し、正解率を出力する関数。
        """
        x_train, x_test, y_train, y_test = train_test_split(a, b, test_size=0.3, shuffle=True, random_state=3, stratify=b)
        model = svm.SVC(C=100, kernel='poly', gamma=0.001)
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        accuracy_train = accuracy_score(y_train, pred_train)
        print('トレーニングデータに対する正解率: %.2f' % accuracy_train)

        pred_test = model.predict(x_test)
        accuracy_test = accuracy_score(y_test, pred_test)
        print('テストデータに対する正解率: %.2f' % accuracy_test)

    @classmethod
    def test_svc_learning(cls):
        """
        svc_learning 関数のテストを行うメソッド。正解率が出力されることを確認する。
        """
        x, y = cls.generate_test_data()
        cls.svc_learning(x, y)

    @staticmethod
    def svc_gridsearch(a, b):
        """
        SVCのグリッドサーチを行い、最適なパラメータを出力する関数。
        """
        params = [
            {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
            {"C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma": [0.001, 0.0001]},
            {"C": [1, 10, 100, 1000], "kernel": ["poly"], "gamma": [0.001, 0.0001]},
            {"C": [1, 10, 100, 1000], "kernel": ["sigmoid"], "gamma": [0.001, 0.0001]},
        ]

        x_train, x_test, y_train, y_test = train_test_split(a, b, test_size=0.3, shuffle=True, random_state=3, stratify=b)

        clf = GridSearchCV(svm.SVC(), params, cv=3)
        clf.fit(x_train, y_train)

        print("最適な学習モデル: ", clf.best_estimator_)

        pre = clf.predict(x_test)
        ac_score = metrics.accuracy_score(y_test, pre)
        print("テストデータに対する正解率: ", ac_score)

    @classmethod
    def test_svc_gridsearch(cls):
        """
        svc_gridsearch 関数のテストを行うメソッド。
        """
        x, y = cls.generate_test_data()
        cls.svc_gridsearch(x, y)

    @staticmethod
    def plot_validation_curve(a, b):
        """
        検証曲線を描画するメソッド。
        """
        param_range = [1, 10, 100, 1000]
        train_scores, test_scores = validation_curve(
            svm.SVC(kernel='poly'), a, b, param_name="C", param_range=param_range, cv=3, scoring="accuracy", n_jobs=-1
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Validation Curve with SVM (poly kernel)")
        plt.xlabel("C")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        plt.xscale("log")
        plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=2)
        plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=2)
        plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=2)
        plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=2)
        plt.legend(loc="best")

    @classmethod
    def test_plot_validation_curve(cls):
        """
        plot_validation_curve 関数のテストを行うメソッド。
        """
        x, y = cls.generate_test_data()
        cls.plot_validation_curve(x, y)
        plt.show()

if __name__ == "__main__":
    print("svc_learningのユニットテスト")
    SVCTest.test_svc_learning()
    #print("svc_gridsearchのユニットテスト")
    #SVCTest.test_svc_gridsearch()
    #print("plot_validation_curveのユニットテスト")
    #SVCTest.test_plot_validation_curve()
    