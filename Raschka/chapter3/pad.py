import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score
from MachineLearning.Raschka.chapter3 import learn_visualization as vis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
iris=datasets.load_iris()
X=iris.data[:,[2,3]]
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

np.random.seed(0)
X_xor=np.random.randn(200,2)
y_xor=np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
y_xor=np.where(y_xor,1,-1)
## p54
# ppn=Perceptron(n_iter=40,eta0=0.1,random_state=0)
# ppn.fit(X_train_std,y_train)
# y_pred=ppn.predict(X_test_std)
# print('Misclassified samples : %d' % (y_test != y_pred).sum())
# print('Accuracy : %.2f' % accuracy_score(y_test,y_pred))
# vis.plot_decision_regions(X=X_combined_std,y=y_combined,classifier=ppn,test_idx=range(105,150))
# plt.xlabel('petal length[cm]')
# plt.ylabel('petal width[cm]')
# plt.legend(loc='upper left')
# plt.show()
## p58
# def sigmoid(z):
#     return 1.0/(1.0+np.exp(-z))
# z=np.arange(-7,7,0.1)
# phi_z=sigmoid(z)
# plt.plot(z,phi_z)
# plt.axvline(0.0,color='k')
# plt.axhspan(0.0,1.0,facecolor='1.0',alpha=1.0,ls='dotted')
# plt.axhline(y=0.5,ls='dotted',color='k')
# plt.yticks([0.0,0.5,1.0])
# plt.ylim(-0.1,1.1)
# plt.xlabel('z')
# plt.ylabel('$\phi (z)$')
# plt.show()
## p62
# lr=LogisticRegression(C=1000.0,random_state=0)
# lr.fit(X_train_std,y_train)
# vis.plot_decision_regions(X_combined_std,y_combined,classifier=lr,test_idx=range(105,150))
# plt.xlabel('petal length[standardized]')
# plt.ylabel('petal width[standardized]')
# plt.legend(loc='upper left')
# plt.show()
## p67
# weights,params=[],[]
# for c in np.arange(-5,5):
#     lr=LogisticRegression(C=10.0**c,random_state=0)
#     lr.fit(X_train_std,y_train)
#     weights.append(lr.coef_[1])
#     params.append(10.0**c)
# weights=np.array(weights)
# plt.plot(params,weights[:,0],label='petal length')
# plt.plot(params,weights[:,1],linestyle='--',label='petal width')
# plt.ylabel('weight coefficient')
# plt.xlabel('C')
# plt.legend(loc='upper left')
# plt.xscale('log')
# plt.show()
## p71
# svm=SVC(kernel='linear',C=1.0,random_state=0)
# svm.fit(X_train_std,y_train)
# vis.plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
# plt.xlabel('petal length[standardized]')
# plt.ylabel('petal width[standardized]')
# plt.legend(loc='upper left')
# plt.show()
## p74
# plt.scatter(X_xor[y_xor==1,0],X_xor[y_xor==1,1],c='b',marker='x',label='1')
# plt.scatter(X_xor[y_xor==-1,0],X_xor[y_xor==-1,1],c='r',marker='s',label='-1')
# plt.ylim(-3.0)
# plt.legend()
# plt.show()
## p76
# svm=SVC(kernel='rbf',random_state=0,gamma=0.1,C=10.0)
# svm.fit(X_xor,y_xor)
# vis.plot_decision_regions(X_xor,y_xor,classifier=svm)
# plt.show()
## p77
# svm=SVC(kernel='rbf',random_state=0,gamma=0.2,C=1.0)
# svm.fit(X_train_std,y_train)
# vis.plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
# plt.xlabel('petal length[standardized]')
# plt.ylabel('petal width[standardized]')
# plt.legend(loc='upper left')
# plt.show()
## p78
# svm=SVC(kernel='rbf',random_state=0,gamma=100.0,C=1.0)
# svm.fit(X_train_std,y_train)
# vis.plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
# plt.xlabel('petal length[standardized]')
# plt.ylabel('petal width[standardized]')
# plt.legend(loc='upper left')
# plt.show()
## p84
# def gini(p):
#     return p*(1-p)+(1-p)*(1-(1-p))
# def entropy(p):
#     return -p*np.log2(p)-(1-p)*np.log2(1-p)
# def error(p):
#     return 1-np.max([p,1-p])
# x=np.arange(0.0,1.0,0.01)
# ent=[entropy(p) if p!=0 else None for p in x]
# sc_ent=[e*0.5 if e else None for e in ent]
# err=[error(i) for i in x]
# fig =plt.figure()
# ax=plt.subplot(111)
# for i,lab,ls,c, in zip([ent,sc_ent,gini(x),err],['Entropy','Entropy(scaled)','Gini Impurity','Misclassification error'],
#                        ['-','-','--','-.'],['black','lightgray','red','green','cyan']):
#     line=ax.plot(x,i,label=lab,linestyle=ls,lw=2,color=c)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,shadow=False)
# ax.axhline(y=0.5,linewidth=1,color='k',linestyle='--')
# ax.axhline(y=1.0,linewidth=1,color='k',linestyle='--')
# plt.ylim([0,1.1])
# plt.xlabel('p(i=1)')
# plt.ylabel('Impurity Index')
# plt.show()
## p85
tree = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
tree.fit(X_train,y_train)
X_combined=np.vstack((X_train,X_test))
y_combined=np.hstack((y_train,y_test))
# vis.plot_decision_regions(X_combined,y_combined,classifier=tree,test_idx=range(105,150))
# plt.xlabel('petal length[standardized]')
# plt.ylabel('petal width[standardized]')
# plt.legend(loc='upper left')
# # plt.show()
export_graphviz(tree,out_file='tree.dot',feature_names=['petal length','petal width'])
## p89
# forest= RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=1,n_jobs=2)
# forest.fit(X_train,y_train)
# vis.plot_decision_regions(X_combined,y_combined,classifier=forest,test_idx=range(105,150))
# plt.xlabel('petal length')
# plt.ylabel('petal width')
# plt.legend(loc='upper left')
# plt.show()
knn=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
knn.fit(X_train_std,y_train)
vis.plot_decision_regions(X_combined_std,y_combined,classifier=knn,test_idx=range(105,150))
plt.xlabel('petal length[standardized]')
plt.ylabel('petal width[standardized]')
plt.legend(loc='upper left')
plt.show()
