import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_and_evaluate_rf_model(data_path):
    # データの読み込み
    df = pd.read_csv(data_path)

    # 特徴量とターゲットに分離
    x = df.iloc[:, :-1]  # 最後の1列以外を取り出す
    y = df.iloc[:, -1]   # 最後の1列のみを取り出す

    # トレーニングデータとテストデータに分割（80% トレーニング、20% テスト）
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # ランダムフォレストモデルの作成
    rf_model = RandomForestClassifier(n_estimators=180, max_depth=None, max_leaf_nodes=None, min_samples_split=4)

    # モデルの訓練
    rf_model.fit(x_train, y_train)

    # テストデータで予測
    y_pred = rf_model.predict(x_test)

    # モデルの評価
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # 結果の表示
    print("Accuracy:", accuracy)
    print("Classification Report:\n", class_report)

    return accuracy, class_report

# 関数の呼び出し
data_path = "winequality-white-re.csv"
train_and_evaluate_rf_model(data_path)
