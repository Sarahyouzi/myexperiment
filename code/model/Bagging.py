# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:31:58 2019

@author: wanru
"""
import scipy
import numpy 
import matplotlib
import pandas
from pandas import read_csv
from matplotlib import pyplot
from pandas import set_option
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
filename='D:\\data\\tzj31.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:2149]
Y=array[:,2149]
filename='D:\\data\\data21_4_1.csv'
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
X=array[:,0:977]
Y=array[:,977]
filename='D:\\data\\data21_5_3.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:977]
Y=array[:,977]
filename='D:\\data\\data2_5_4.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:977]
Y=array[:,977]

filename='D:\\data\\add1.csv'
dataset=read_csv(filename,header=None)
array=dataset.values
X=array[:,0:1]
Y=array[:,1]


num_folds=5
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)
cart=DecisionTreeClassifier()
cart=SVC(kernel='rbf', probability=True,C=2.639,gamma=0.035879)
model=BaggingClassifier(base_estimator=cart,n_estimators=800,random_state=seed)
result=cross_val_score(model,X,Y,cv=kfold)
print(result.mean())




#参数寻优
#设置要遍历的参数
#svm
scoring = 'accuracy'
param_grid = {}
param_grid['n_estimators'] = [10,50, 100, 200, 300,400, 500, 600,700,800,900]
cart=SVC(kernel='rbf', probability=True,C=0.87055,gamma=0.00097656)
model=BaggingClassifier(base_estimator=cart,random_state=seed)

kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X,Y)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))
    
    
#dt
#参数寻优
#设置要遍历的参数
scoring = 'accuracy'
param_grid = {}
param_grid['n_estimators'] = [10,50, 100, 200, 300,400, 500, 600,700,800,900]
cart=DecisionTreeClassifier(max_depth=2,min_samples_split=2)

model=BaggingClassifier(base_estimator=cart,random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X,Y)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
s='最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_)
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))
    

#LR   
scoring = 'accuracy'
param_grid = {}
param_grid['n_estimators'] = [10,50, 100, 200, 300,400, 500, 600,700,800,900]
cart=LogisticRegression(C=0.1,solver='newton-cg')
model=BaggingClassifier(base_estimator=cart)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X,Y)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
s='最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_)
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))
 #bayes    
 
# lda = LinearDiscriminantAnalysis(n_components=8)
#lda.fit(X,Y)
#X = lda.transform(X)
 
scoring = 'accuracy'
param_grid = {}
param_grid['n_estimators'] = [10,50, 100, 200, 300,400, 500, 600,700,800,900]
cart=GaussianNB()
model=BaggingClassifier(base_estimator=cart,random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X,Y)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
s='最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_)
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
        cart=SVC(kernel='rbf', probability=True,C=0.87055,gamma=0.00097656)
        model=BaggingClassifier(base_estimator=cart,random_state=seed,n_estimators=10)
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
    c[i,3]=Acc
    
    
    
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
res=mat(zeros((10,1)))
for i in r1:
     # x：数据
    # y：类别标签
    # pct：训练集所占比例
    cart=SVC(kernel='rbf', probability=True,C=2.2974,gamma=0.020617)	
#    model=BaggingClassifier(base_estimator=cart,random_state=seed,n_estimators=300)
#    
    cart=SVC(kernel='rbf', probability=True,C=3.0314,gamma=0.0034006)	
    model=BaggingClassifier(base_estimator=cart,random_state=seed,n_estimators=50)
    model.fit(X,Y)
#    filename='D:\\data\\terE4.2.csv'
#    filename='D:\\data\\terB6.2.csv'
#    filename='D:\\data\\terB5.31.csv'
    filename='D:\\data\\terE4.11.csv'
#    filename='D:\\data\\terBadd1.csv'
    dataset=read_csv(filename,header=None)
    array=dataset.values
    X_test=array[:,0:256]
    Y_test=array[:,256]
#    lda = LinearDiscriminantAnalysis(n_components=5)
#    lda.fit(X_test,Y_test)
#    X_test = lda.transform(X_test)
    result=model.score(X_test,Y_test)
    res[i,0]=result

a=np.mean(res[:,0])
s=np.std(res[:,0], ddof=1)
d=s/np.sqrt(10)
print(a)
print(d)