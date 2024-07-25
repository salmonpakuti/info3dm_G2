from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
import numpy as np
import pandas as pd

df = pd.read_csv('winequality-white-re.csv')


x = df.iloc[:,0:10]
x=pd.DataFrame(df[["residual sugar","free sulfur dioxide","total sulfur dioxide","alcohol"]])
y = df.iloc[:,11]

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size= 0.7,random_state=0,stratify=y)

model = svm.SVC(C=100,kernel='poly',gamma=0.001)

model.fit(x_train,y_train)

pred_train = model.predict(x_train)
accuracy_train = accuracy_score(y_train,pred_train)
print('トレーニングデータに対する正解率:%.2f'% accuracy_train)

pred_test = model.predict(x_test)
accuracy_test = accuracy_score(y_test , pred_test)
print('テストデータに対する正解率:%.2f'% accuracy_test)