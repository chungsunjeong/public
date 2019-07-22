'''
Chapter6.1-4 : Model Evaluation and Cross-validation
From p173
To p193
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
X=df.loc[:,2:].values
y=df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

## Pipeline
## p175
# pipe_lr = Pipeline([('scl', StandardScaler()),
#                     ('pca', PCA(n_components=2)),
#                     ('clf', LogisticRegression(random_state=1))])
# pipe_lr.fit(X_train, y_train)
# print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
# y_pred = pipe_lr.predict(X_test)

## k-fold crossvalidation
from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(n_splits=10,random_state=1).split(X_train, y_train)
# scores=[]
# for k, (train, test) in enumerate(kfold):
#     pipe_lr.fit(X_train[train], y_train[train])
#     score = pipe_lr.score(X_train[test], y_train[test])
#     scores.append(score)
#     print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k + 1,np.bincount(y_train[train]), score))
# print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

## k-fold crossvalidation lib
from sklearn.model_selection import cross_val_score
# scores = cross_val_score(estimator=pipe_lr,
#                          X=X_train,
#                          y=y_train,
#                          cv=10,
#                          n_jobs=1)
# print('CV accuracy scores: %s' % scores)
# print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

## Diagnosing bias and variance problems with learning curves
## p185
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
# pipe_lr = Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(penalty='l2', random_state=0))])
# train_sizes, train_scores, test_scores =learning_curve(estimator=pipe_lr,
#                                X=X_train,
#                                y=y_train,
#                                train_sizes=np.linspace(0.1, 1.0, 10),
#                                cv=10,
#                                n_jobs=1)
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
# plt.plot(train_sizes, train_mean,color='blue', marker='o',markersize=5, label='training accuracy')
# plt.fill_between(train_sizes,train_mean + train_std,train_mean - train_std,alpha=0.15, color='blue')
# plt.plot(train_sizes, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy')
# plt.fill_between(train_sizes,test_mean + test_std,test_mean - test_std,alpha=0.15, color='green')
# plt.grid()
# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.ylim([0.8, 1.0])
# plt.tight_layout()
# # plt.savefig('./figures/learning_curve.png', dpi=300)
# plt.show()

## Addressing over- and underfitting with validation curves
## p187
from sklearn.model_selection import validation_curve
# param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# train_scores, test_scores = validation_curve(
#                 estimator=pipe_lr,
#                 X=X_train,
#                 y=y_train,
#                 param_name='clf__C',
#                 param_range=param_range,
#                 cv=10)
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
# plt.plot(param_range, train_mean,color='blue', marker='o',markersize=5, label='training accuracy')
# plt.fill_between(param_range, train_mean + train_std,train_mean - train_std, alpha=0.15,color='blue')
# plt.plot(param_range, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy')
# plt.fill_between(param_range,test_mean + test_std,test_mean - test_std,alpha=0.15, color='green')
# plt.grid()
# plt.xscale('log')
# plt.legend(loc='lower right')
# plt.xlabel('Parameter C')
# plt.ylabel('Accuracy')
# plt.ylim([0.8, 1.0])
# plt.tight_layout()
# # plt.savefig('./figures/validation_curve.png', dpi=300)
# plt.show()



from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

## Tuning hyperparameters via grid search
## p189
pipe_svc = Pipeline([('scl', StandardScaler()),
            ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
                 {'clf__C': param_range,
                  'clf__gamma': param_range,
                  'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)
# if __name__ == '__main__':
#     gs = gs.fit(X_train, y_train)
#     print(gs.best_score_)
#     print(gs.best_params_)
#
#     clf=gs.best_estimator_
#     clf.fit(X_train, y_train)
#     print('Test accuracy: %.3f' % clf.score(X_test, y_test))

## Algorithm selection with nested cross-validation
    # scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    # print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))