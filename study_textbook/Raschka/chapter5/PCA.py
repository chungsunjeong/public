'''
Chapter5.1 : Principal Component Analysis practice
From p129
To p141
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from MachineLearning.Raschka.chapter3.learn_visualization import plot_decision_regions

## p132
df_wine=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
X,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.fit_transform(X_test)

## p133
# cov_mat=np.cov(X_train_std.T)
# eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
# print(eigen_vals,eigen_vecs)

## p134
# tot=sum(eigen_vals)
# var_exp=[(i/tot) for i in sorted(eigen_vals,reverse=True)]
# cum_var_exp=np.cumsum(var_exp)
# plt.bar(range(1,14),var_exp,alpha=0.5,align='center',label='individual explained variance')
# plt.step(range(1,14),cum_var_exp,where='mid',label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal Components')
# plt.legend(loc='best')
# plt.show()

## p135
# eigen_pairs=[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
# eigen_pairs.sort(reverse=True)
# print(eigen_pairs)
# w=np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
# print('Matrix W:\n',w)
# X_train_pca=X_train_std.dot(w)

## p137
# colors = ['r', 'b', 'g']
# markers = ['s', 'x', 'o']
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_pca[y_train == l, 0],
#                 X_train_pca[y_train == l, 1],
#                 c=c, label=l, marker=m)
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# plt.show()

## Using PCA lib
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# pca=PCA(n_components=2)
# lr=LogisticRegression()
# X_train_pca=pca.fit_transform(X_train_std)
# X_test_pca=pca.transform(X_test_std)
# lr.fit(X_train_pca,y_train)
# plot_decision_regions(X_train_pca,y_train,classifier=lr)
# plot_decision_regions(X_test_pca,y_test,classifier=lr)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend(loc='lower left')
# plt.show()

## p140
# pca=PCA(n_components=None)
# X_train_pca=pca.fit_transform(X_train_std)
# print(pca.explained_variance_ratio_)