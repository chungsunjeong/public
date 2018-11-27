import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from MachineLearning.Raschka.chapter2 import learn ,learn_visualization as vis
# Data load
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
# print(df.iloc[0:100])
y_total=df.iloc[0:150,4].values
y_total=np.where(y_total=='Iris-setosa',-1,1)
X_total=df.iloc[0:150,[0,2]].values
y=y_total[0:100]
X=X_total[0:100,0:]
ppn=learn.Perceptron(eta=0.01,n_iter=10)

## p29
# plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
# plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
# plt.scatter(X[100:150,0],X[100:150,1],color='green',marker='o',label='virginica')
# plt.xlabel('petal length')
# plt.ylabel('sepal length')
# plt.legend(loc='upper left')
# plt.show()
## p30
# ppn.fit(X,y)
# plt.plot(range(1,ppn.n_iter+1),ppn.errors_,marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of misclassification')
# plt.grid()
# plt.show()
## p32
# ppn.fit(X,y)
# vis.plot_decision_regions(X,y,classifier=ppn)
# plt.xlabel('sepal length[cm]')
# plt.ylabel('petal length[cm]')
# plt.legend(loc='upper left')
# plt.show()
## p39
# fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(8,4))
# ada1=learn.AdalineGD(n_iter=10,eta=0.01).fit(X,y)
# ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('log(Sum-squared-error)')
# ax[0].set_title('Adaline-Learning rate 0.01')
# ada2=learn.AdalineGD(n_iter=10,eta=0.0001).fit(X,y)
# ax[1].plot(range(1,len(ada2.cost_)+1),np.log10(ada2.cost_),marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('log(Sum-squared-error)')
# ax[1].set_title('Adaline-Learning rate 0.0001')
# plt.show()
## p41
# X_std=np.copy(X)
# X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
# X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()
# ada=learn.AdalineGD(n_iter=15,eta=0.01)
# ada.fit(X_std,y)
# vis.plot_decision_regions(X_std,y,classifier=ada)
# plt.title('Adaline - GD')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.legend(loc='upper left')
# plt.show()
# plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Sum-squared-error')
# plt.show()
## p47
X_std=np.copy(X)
X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()
ada=learn.AdalineSGD(n_iter=15,eta=0.01,random_state=1)
ada.fit(X_std,y)
vis.plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline - SGD')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
