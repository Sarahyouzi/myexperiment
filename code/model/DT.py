# -*- coding: utf-8 -*-
"""
Created on Mon Jan 2 16:33:36 2019

@author: wanru
"""
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import scipy
from numpy import *
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from pandas import set_option
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


filename='D:\\data\\pwm161.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:2]
Y=array[:,2]

filename='D:\\data\\tzj31.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:2149]
Y=array[:,2149]

filename='D:\\data\\data2_4_1.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:256]
Y=array[:,256]

filename='D:\\data\\data2_4_2.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:248]
Y=array[:,248]



filename='D:\\data\\data2_5_3.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:1024]
Y=array[:,1024]

filename='D:\\data\\data2_5_4.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:977]
Y=array[:,977]

filename='D:\\data\\data13.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:258]
Y=array[:,258]
filename='D:\\data\\data23.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:255]
Y=array[:,255]

filename='D:\\data\\data34.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:977]
Y=array[:,977]

filename='D:\\data\\data44.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:1012]
Y=array[:,1012]

filename='D:\\data\\add1.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:1]
Y=array[:,1]

num_folds=5
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)
model=DecisionTreeClassifier()
result=cross_val_score(model,X,Y,cv=kfold)
print(result.mean())


scoring = 'accuracy'
#设置参数矩阵：
param_grid = {}
param_grid['min_samples_split'] = np.arange(2,30,2)
param_grid['max_depth'] = np.arange(2,10)
model=DecisionTreeClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X,Y)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))




r1=np.arange(0,10,1)
r2=np.arange(0,5,1)
c=mat(ones((10,4)))
res=mat(ones((10,1)))
for i in r1:
     # x：数据
    # y：类别标签
    # pct：训练集所占比例
    length = len(Y)
    rr=np.arange(0,length,1)
    index = round(length/5)
    indexes =np.random.choice(rr,length)
    data_X=X[indexes[0:length]]
    data_Y=Y[indexes[0:length]]
    n1=0
    n0=0
    n10=0
    n01=0
    for j in r2:

        if j==0 :
            X_test=data_X[0:index]
            Y_test=data_Y[0:index]
            X_train=data_X[index-1:length]
            Y_train=data_Y[index-1:length]
        elif j==4:
            X_test=data_X[index*j-1:index*(j+1)]
            Y_test=data_Y[index*j-1:index*(j+1)]
            X_train=data_X[0:index*(j+1)]
            Y_train=data_Y[0:index*(j+1)]
        else:
            #    random.shuffle(indexes)
            X_test=data_X[index*j-1:index*(j+1)-1]
            Y_test=data_Y[index*j-1:index*(j+1)-1]
            X_train1=data_X[0:index*j-1]
            Y_train1=data_Y[0:index*j-1]
            X_train2=data_X[index*(j+1)-1:length]
            Y_train2=data_Y[index*(j+1)-1:length]
            X_train=np.row_stack((X_train1,X_train2))
            Y_train=np.concatenate((Y_train1,Y_train2),axis=0)
        model =DecisionTreeClassifier(max_depth=2,min_samples_split=2)

        model.fit(X_train,Y_train)
        predicted=model.predict(X_test)
        matrix=confusion_matrix(Y_test,predicted)
        classes=['0','1']
        dataframe=pd.DataFrame(data=matrix,index=classes,columns=classes)
    #print(dataframe)
        da=dataframe.values
        n1=n1+da[1,0]+da[1,1]
        n0=n0+da[0,0]+da[0,1]
        n10=n10+da[1,0]
        n01=n01+da[0,1]
        Sn=1-n10/n1
        Sp=1-n01/n0
        Acc=1-(n10+n01)/(n1+n0)
        Mcc=(1-n10/n1-n01/n0)/((1+(n01-n10)/n1)*(1+(n10-n01)/n0)**0.5)
    c[i,0]=Sn
    c[i,1]=Sp
    c[i,2]=Mcc
    c[i,3]=Acc*100

a=np.mean(c[:,0])
s=np.std(c[:,0], ddof=1)
d=s/np.sqrt(10)
print(a)
print(d)

a=np.mean(c[:,1])
s=np.std(c[:,1], ddof=1)
d=s/np.sqrt(10)
print(a)
print(d)

a=np.mean(c[:,2])
s=np.std(c[:,2], ddof=1)
d=s/np.sqrt(10)
print(a)
print(d)

a=np.mean(c[:,3])
s=np.std(c[:,3], ddof=1)
d=s/np.sqrt(10)
print(a)
print(d)


r1=np.arange(0,10,1)
res=mat(ones((10,1)))
for i in r1:
     # x：数据
    # y：类别标签
    # pct：训练集所占比例
    length = len(Y)
    rr=np.arange(0,length,1)
    index = round(length*0.8)
    indexes =np.random.choice(rr,length)
#    random.shuffle(indexes)
    trn_idxes = indexes[0:index]
    X_train = X[trn_idxes,:]
    Y_train = Y[trn_idxes,]

    model =DecisionTreeClassifier()
    model.fit(X_train,Y_train)
#    filename='D:\\data\\terE4.1.csv'
#    filename='D:\\data\\terB4.1.csv'
#    filename='D:\\data\\terB6.2.csv'
    filename='D:\\data\\terB4.2.csv'
    dataset=read_csv(filename,header=None)
    X_test=dataset.values
    filename='D:\\data\\terB.csv'
#    filename='D:\\data\\terB.csv'
    dataset=read_csv(filename,header=None)
    Y_test=dataset.values
    result=model.score(X_test,Y_test)
    res[i,0]=result*100
max=res.max(0)
min=res.min(0)
ave=(max+min)/2
value=max-ave
print(ave)
print(value)

