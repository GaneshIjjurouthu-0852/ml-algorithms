from knn_scratch import KNNClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data=pd.read_csv("Iris.csv")
X_2d=data[['PetalLengthCm','PetalWidthCm']].values
y_2d=data['Species'].values
X_train_2d,X_test_2d,y_train_2d,y_test_2d=train_test_split(X_2d,y_2d,test_size=0.2,random_state=42)
scaler_2d=StandardScaler()
X_train_2d=scaler_2d.fit_transform(X_train_2d)
X_test_2d=scaler_2d.transform(X_test_2d)
plt.figure(figsize=(10,7))

x_min,x_max=X_test_2d[:,0].min() - 1,X_test_2d[:,0].max() + 1
y_min,y_max=X_test_2d[:,1].min() - 1,X_test_2d[:,1].max() + 1 

xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
grid_points=np.c_[xx.ravel(),yy.ravel()]

knn_2d=KNNClassifier(k=3)
knn_2d.fit(X_train_2d,y_train_2d)
Z=knn_2d.predict(grid_points)

label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
Z_num=np.array([label_map[z] for z in Z])
Z_num=Z_num.reshape(xx.shape)
plt.contourf(xx, yy, Z_num, alpha=0.3, cmap='RdYlGn')
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
for label, color in colors.items():
    idx = y_test_2d == label     
    plt.scatter(X_test_2d[idx, 0], X_test_2d[idx, 1],
                color=color, label=label, edgecolors='black', s=80)

plt.xlabel('Petal Length (scaled)')
plt.ylabel('Petal Width (scaled)')
plt.title('KNN Decision Boundary (k=3)')
plt.legend()
plt.show()