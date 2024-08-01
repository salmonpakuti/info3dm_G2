import unittest
import pandas as pd
from randomforest_experiment import train_and_evaluate_rf_model

class TestRFModel(unittest.TestCase):
    def test_train_and_evaluate_rf_model(self):
        df = "winequality-white-re.csv"

        # 関数の呼び出し
        accuracy, class_report = train_and_evaluate_rf_model(df)

        # 精度と分類レポートが表示されることを確認
        self.assertIsInstance(accuracy, float)
    
        self.assertIsInstance(class_report, str)
        #self.assertGreater(len(class_report), 0)

if __name__ == '__main__':
    unittest.main()