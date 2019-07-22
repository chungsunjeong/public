'''
Chapter4 : Data Preprocessing
From p107
To End of Ch.4
'''
import pandas as pd
import numpy as np
df_wine=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns=['Class label','Alcohol','Malic acid','Ash','Alcalinity of ash',
                 'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins',
                 'Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
## p108
# print('Class labels',np.unique(df_wine['Class label']))
# print(df_wine.head(10))
## Partition to train and test data set
from sklearn.model_selection import train_test_split
X,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
## p110
## MMS Scaling
# from sklearn.preprocessing import MinMaxScaler
# mms=MinMaxScaler()
# X_train_norm=mms.fit_transform(X_train)
# X_test_norm=mms.fit_transform(X_test)
## Standard Scaling
from sklearn.preprocessing import StandardScaler
stdsc=StandardScaler()
X_train_std=stdsc.fit_transform(X_train)
X_test_std=stdsc.fit_transform(X_test)
## p115
## L1 Regularization
from sklearn.linear_model import LogisticRegression
# lr=LogisticRegression(penalty='l1',C=0.1)
# lr.fit(X_train_std,y_train)
# print('Training accuracy:', lr.score(X_train_std,y_train))
# print('Test accuracy:', lr.score(X_test_std,y_test))
# print(lr.intercept_)
# print(lr.coef_)

## p117
import matplotlib.pyplot as plt
# fig=plt.figure()
# ax=plt.subplot(111)
# colors=['blue','green','red','cyan','magenta','yellow','black','pink','lightgreen','lightblue','gray','indigo','orange']
# weights,params=[],[]
# for c in np.arange(-4,6):
#     lr=LogisticRegression(penalty='l1',C=10.0**c,random_state=0)
#     lr.fit(X_train_std,y_train)
#     weights.append(lr.coef_[1])
#     params.append(10.0**c)
# weights=np.array(weights)
# for column,color in zip(range(weights.shape[1]),colors):
#     plt.plot(params,weights[:,column],label=df_wine.columns[column+1],color=color)
# plt.axhline(0,color='black',linestyle='--',linewidth=3)
# plt.xlim([10.0**(-5),10.0**5])
# plt.ylabel('weight coefficient')
# plt.xlabel('C')
# plt.xscale('log')
# plt.legend(loc='upper left')
# ax.legend(loc='upper center',bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=True)
# plt.show()

## Sequential Backward Selection
## 121
from sklearn.neighbors import KNeighborsClassifier
from MachineLearning.Raschka.chapter4.SequentialBackwardSelection import SBS
knn=KNeighborsClassifier(n_neighbors=2)
# sbs=SBS(knn,k_features=1)
# sbs.fit(X_train_std,y_train)
# k_feat=[len(k) for k in sbs.subsets_]
# plt.plot(k_feat,sbs.scores_,marker='o')
# plt.ylim([0.7,1.1])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
# plt.show()
# k5=list(sbs.subsets_[8])
# print(df_wine.columns[1:][k5])
## Feature reduction through SBS
## 123
# knn.fit(X_train_std,y_train)
# print('original KNN')
# print('Training accuracy:',knn.score(X_train_std,y_train))
# print('Test accuracy:',knn.score(X_test_std,y_test))
# knn.fit(X_train_std[:,k5],y_train)
# print('KNN with SBS')
# print('Training accuracy:',knn.score(X_train_std[:,k5],y_train))
# print('Test accuracy:',knn.score(X_test_std[:,k5],y_test))
## Feature reduction through Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
feat_labels=df_wine.columns[1:]
forest=RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
forest.fit(X_train,y_train)
importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]
# for f in range(X_train.shape[1]):
#     print("%2d) %-*s %f" %(f+1,30,feat_labels[f],importances[indices[f]]))
#
# plt.title('Feature Importances')
# plt.bar(range(X_train.shape[1]),importances[indices],color='lightblue',align='center')
# plt.xticks(range(X_train.shape[1]),feat_labels,rotation=90)
# plt.xlim([-1,X_train.shape[1]])
# plt.tight_layout()
# plt.show()
sfm=SelectFromModel(forest,threshold=0.15,prefit=True)
X_selected=sfm.transform(X_train)
print(X_selected.shape)
for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))