# KNN from Scratch
import pandas as pd

data=pd.read_csv("Iris.csv")
data=data.drop(columns=['Id'])
X=data.iloc[:, :-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

import numpy as np
from collections import Counter
class KNNClassifier():
    def __init__(self,k):
        self.k=k
    def fit(self,X,y):
        self.X=X
        self.y=y
    def predict(self,X):
        predictions=[]
        for x in X:
            result=self._predict_one(x)
            predictions.append(result)
        return np.array(predictions)
    def _predict_one(self,x):
        distances=np.linalg.norm(self.X-x,axis=1)
        k_indices=np.argsort(distances)[:self.k]
        k_labels=self.y[k_indices]
        return Counter(k_labels).most_common(1)[0][0]

knn=KNNClassifier(k=3)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy: ",accuracy_score(y_pred,y_test))

import matplotlib.pyplot as plt
X_2d=data[['PetalLengthCm','PetalWidthCm']].values
y_2d=data['Species'].values
X_train_2d,X_test_2d,y_train_2d,y_test_2d=train_test_split(X_2d,y_2d,test_size=0.2,random_state=42)
scaler_2d=StandardScaler()
X_train_2d=scaler_2d.fit_transform(X_train_2d)
X_test_2d=scaler_2d.transform(X_test_2d)
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
knn_2d=KNNClassifier(k=3)
knn_2d.fit(X_train_2d,y_train_2d)
y_pred_2d=knn_2d.predict(X_test_2d)
plt.figure(figsize=(8,6))
for label,color in colors.items():
    idx=y_pred_2d==label
    plt.scatter(X_test_2d[idx,0],X_test_2d[idx,1],color=color,label=label,edgecolors='black',s=80)
plt.xlabel('Petal Length (scaled)')
plt.ylabel('Petal Width (scaled)')
plt.title('KNN Predictions on Test Set')
plt.legend()
plt.show()


