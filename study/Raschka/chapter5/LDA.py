'''
Chapter5.2 : Linear Discriminant Analysis practice
From p141
To p151
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from MachineLearning.Raschka.chapter3.learn_visualization import plot_decision_regions

df_wine=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
X,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.fit_transform(X_test)

## p143
## Scatter Matrix within class
# np.set_printoptions(precision=4)
# mean_vecs = []
# for label in range(1, 4):
#     mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
#     print('MV %s: %s\n' % (label, mean_vecs[label - 1]))
#
# d = 13  # number of features
# S_W = np.zeros((d, d))
# for label, mv in zip(range(1, 4), mean_vecs):
#     class_scatter = np.zeros((d, d))  # scatter matrix for each class
#     for row in X_train_std[y_train == label]:
#         row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
#         class_scatter += (row - mv).dot((row - mv).T)
#     S_W += class_scatter                          # sum class scatter matrices

# print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
# print('Class label distribution: %s' % np.bincount(y_train)[1:])

# d = 13  # number of features
# S_W = np.zeros((d, d))
# for label, mv in zip(range(1, 4), mean_vecs):
#     class_scatter = np.cov(X_train_std[y_train == label].T)
#     S_W += class_scatter
# print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

## p145
## Scatter Matrix between class
# mean_overall = np.mean(X_train_std, axis=0)
# d = 13  # number of features
# S_B = np.zeros((d, d))
# for i, mean_vec in enumerate(mean_vecs):
#     n = X_train[y_train == i + 1, :].shape[0]
#     mean_vec = mean_vec.reshape(d, 1)  # make column vector
#     mean_overall = mean_overall.reshape(d, 1)  # make column vector
#     S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

# print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))

## p146
# eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Make a list of (eigenvalue, eigenvector) tuples
# eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
# eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

# print('Eigenvalues in decreasing order:\n')
# for eigen_val in eigen_pairs:
#     print(eigen_val[0])

# tot = sum(eigen_vals.real)
# discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
# cum_discr = np.cumsum(discr)
#
# plt.bar(range(1, 14), discr, alpha=0.5, align='center',
#         label='individual "discriminability"')
# plt.step(range(1, 14), cum_discr, where='mid',
#          label='cumulative "discriminability"')
# plt.ylabel('"discriminability" ratio')
# plt.xlabel('Linear Discriminants')
# plt.ylim([-0.1, 1.1])
# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('./figures/lda1.png', dpi=300)
# plt.show()

# w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
#               eigen_pairs[1][1][:, np.newaxis].real))
# print('Matrix W:\n', w)
#
# X_train_lda = X_train_std.dot(w)
# colors = ['r', 'b', 'g']
# markers = ['s', 'x', 'o']
#
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_lda[y_train == l, 0] * (-1),
#                 X_train_lda[y_train == l, 1] * (-1),
#                 c=c, label=l, marker=m)
#
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower right')
# plt.tight_layout()
# # plt.savefig('./figures/lda2.png', dpi=300)
# plt.show()

## Using LDA lib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)

# plot_decision_regions(X_train_lda, y_train, classifier=lr)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# # plt.savefig('./images/lda3.png', dpi=300)
# plt.show()

X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('./images/lda4.png', dpi=300)
plt.show()