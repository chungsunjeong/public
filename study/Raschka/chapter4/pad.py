'''
Chapter4 : Data Preprocessing
From p97
To p107
'''
import pandas as pd
import numpy as np
## p98
# from io import StringIO
# csv_data='''A,B,C,D \n1.0,2.0,3.0,4.0 \n5.0,6.0,,8.0 \n10.0,11.0,12.0,''' #\n,,,'''
# df=pd.read_csv(StringIO(csv_data))
## p98
## NaN remove
# print(df.isnull().sum())
# print('---------------------')
# print(df.dropna())
# print('---------------------')
# print(df.dropna(axis=1))
# print('---------------------')
# print(df.dropna(how='all'))
# print('---------------------')
# print(df.dropna(thresh=3))
# print('---------------------')
# print(df.dropna(subset=['C']))
# print('---------------------')
# print(df.values)
## p100
## NaN imputer
# from sklearn.preprocessing import Imputer
# imr=Imputer(missing_values='NaN',strategy='mean',axis=0)
# imr=imr.fit(df)
# imputed_data=imr.transform(df.values)
# print(imputed_data)
## p103
## Categorical Data Preprocessing
df=pd.DataFrame([['green','M',10.1,'class1'],['red','L',13.5,'class2'],['blue','XL',15.3,'class1']])
df.columns=['color','size','price','classlabel']
print(df)
## Priority Feature Mapping
size_mapping={'XL':3,'L':2,'M':1}
class_mapping={label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
df['size']=df['size'].map(size_mapping)
df['classlabel']=df['classlabel'].map(class_mapping)
# print(df)

## Inverse Mapping
# inv_class_mapping={v:k for k,v in class_mapping.items()}
# df['classlabel']=df['classlabel'].map(inv_class_mapping)
# print(df)

## Another Method of Mapping
from sklearn.preprocessing import LabelEncoder
# class_le=LabelEncoder()
# y=class_le.fit_transform(df['classlabel'].values)
# print(y)
# print(class_le.inverse_transform(y))

## One-hot encoding
from sklearn.preprocessing import OneHotEncoder
X=df[['color','size','price']].values
color_le=LabelEncoder()
X[:,0]=color_le.fit_transform(X[:,0])
print(X)
ohe=OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())
print(pd.get_dummies(df[['price','color','size']]))