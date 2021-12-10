# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 23:38:18 2020

@author: KIIT
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder
def split_age(thresh = 40):
    input_file = r"C:\Users\KIIT\Desktop\Causal\data\diab1.csv"
    input_data = pd.read_csv(input_file)
    input_data = input_data.iloc[:,1:]
    train_data = input_data[input_data.Age <= thresh]
    test_data = input_data[input_data.Age > thresh]
    y_train = train_data['class']
    y_test = test_data['class']
    X_train_data = train_data.drop(['class'], axis =1)
    X_test_data = test_data.drop(['class'], axis =1)
    return X_train_data,y_train, X_test_data, y_test

def split_gender():
    input_file = r"C:\Users\KIIT\Desktop\Causal\data\diab1.csv"
    input_data = pd.read_csv(input_file)
    input_data = input_data.iloc[:,1:]
    train_data = input_data[input_data.Gender == 0]
    test_data = input_data[input_data.Gender == 1]
    y_train = train_data['class']
    y_test = test_data['class']
    X_train_data = train_data.drop(['class'], axis =1)
    X_test_data = test_data.drop(['class'], axis =1)
    return X_train_data,y_train, X_test_data, y_test
    
X_train_data,y_train, X_test_data, y_test = split_age()
input_file = r"C:\Users\KIIT\Desktop\Causal\data\diab1.csv"
input_data = pd.read_csv(input_file)
input_data = input_data.iloc[:,1:]
train_data = input_data[(input_data.Age <= 40 )&( input_data.Gender == 1)]
test_data = input_data[(input_data.Age > 40 )&(input_data.Gender == 0)]
y_train = train_data['class']
y_test = test_data['class']
X_train_data = train_data.drop(['class'], axis =1)
X_test_data = test_data.drop(['class'], axis =1)
    
    
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from skfeature.function.information_theoretical_based import LCSI
from sklearn.ensemble import AdaBoostRegressor
from chefboost import Chefboost as chef

from utils import *
######### Baseline
import time
start = time.time()
n_estimators = 1000
model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
model.fit(X_train_data,y_train)
y_pred = model.predict(X_test_data)

print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)

############ GSS
start = time.time()
ls = ["Polyuria","Polydipsia","Irritability"]
X_train = X_train_data[ls]
X_test = X_test_data[ls]
model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)

########### CMIM-SVR
start = time.time()    
X = X_train_data
y = y_train
n_samples, n_features = X.shape
is_n_selected_features_specified = False

F = np.nan * np.zeros(n_features)

# t1
t1 = np.zeros(n_features)

# m is a counting indicator
m = np.zeros(n_features) - 1

for i in range(n_features):
    f = X.iloc[:, i]
    t1[i] = midd(f, y)



for k in range(n_features):
    if k == 0:
        # select the feature whose mutual information is the largest
        idx = np.argmax(t1)
        F[0] = idx
        f_select = X.iloc[:, idx]

    if is_n_selected_features_specified:
        if np.sum(~np.isnan(F)) == n_selected_features:
            break

    sstar = -1000000 # start with really low value for best partial score sstar 
    for i in range(n_features):
        
        if i not in F:
            
            while (t1[i] > sstar) and (m[i]<k-1) :
                m[i] = m[i] + 1
                t1[i] = min(t1[i], cmidd(X.iloc[:,i], # feature i
                                         y,  # target
                                         X.iloc[:, int(F[int(m[i])])] # conditionned on selected features
                                        )
                           )
            if t1[i] > sstar:
                sstar = t1[i]
                F[k+1] = i
                
F = np.array(F[F>-100])
F = F.astype(int)
t1 = t1[F]

regr = make_pipeline(StandardScaler(),  LinearSVR(random_state=0, tol=1e-3))
regr.fit(X_train_data.iloc[:,F[:10]], y_train)

y_pred = regr.predict(X_test_data.iloc[:,F[:10]])

print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)

##########CMIM-kNNR
start= time.time()
X = X_train_data
y = y_train
n_samples, n_features = X.shape
is_n_selected_features_specified = False

F = np.nan * np.zeros(n_features)

# t1
t1 = np.zeros(n_features)

# m is a counting indicator
m = np.zeros(n_features) - 1

for i in range(n_features):
    f = X.iloc[:, i]
    t1[i] = midd(f, y)

for k in range(n_features):
    ### uncomment to keep track
    # counter = int(np.sum(~np.isnan(F)))
    # if counter%5 == 0 or counter <= 1:
    #     print("F contains %s features"%(counter))
    if k == 0:
        # select the feature whose mutual information is the largest
        idx = np.argmax(t1)
        F[0] = idx
        f_select = X.iloc[:, idx]

    if is_n_selected_features_specified:
        if np.sum(~np.isnan(F)) == n_selected_features:
            break

    sstar = -1000000 # start with really low value for best partial score sstar 
    for i in range(n_features):
        
        if i not in F:
            
            while (t1[i] > sstar) and (m[i]<k-1) :
                m[i] = m[i] + 1
                t1[i] = min(t1[i], cmidd(X.iloc[:,i], # feature i
                                         y,  # target
                                         X.iloc[:, int(F[int(m[i])])] # conditionned on selected features
                                        )
                           )
            if t1[i] > sstar:
                sstar = t1[i]
                F[k+1] = i
                
F = np.array(F[F>-100])
F = F.astype(int)
t1 = t1[F]

error_rate = []
for i in range(1,25):
 knn = KNeighborsRegressor(n_neighbors=i)
 knn.fit(X_train_data.iloc[:,F[:10]],y_train)
 pred_i = knn.predict(X_test_data.iloc[:,F[:10]])
 error_rate.append(np.mean(pred_i != y_test))

knn = KNeighborsRegressor(n_neighbors= error_rate.index(min(error_rate)) if error_rate.index(min(error_rate)) > 0 else 1)
knn.fit(X_train_data.iloc[:,F[:10]],y_train)
y_pred = knn.predict(X_test_data.iloc[:,F[:10]])


print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)

#######################CMIM-RFR

start = time.time()    
X = X_train_data
y = y_train
n_samples, n_features = X.shape
is_n_selected_features_specified = False

F = np.nan * np.zeros(n_features)

# t1
t1 = np.zeros(n_features)

# m is a counting indicator
m = np.zeros(n_features) - 1

for i in range(n_features):
    f = X.iloc[:, i]
    t1[i] = midd(f, y)

for k in range(n_features):
    ### uncomment to keep track
    # counter = int(np.sum(~np.isnan(F)))
    # if counter%5 == 0 or counter <= 1:
    #     print("F contains %s features"%(counter))
    if k == 0:
        # select the feature whose mutual information is the largest
        idx = np.argmax(t1)
        F[0] = idx
        f_select = X.iloc[:, idx]

    if is_n_selected_features_specified:
        if np.sum(~np.isnan(F)) == n_selected_features:
            break

    sstar = -1000000 # start with really low value for best partial score sstar 
    for i in range(n_features):
        
        if i not in F:
            
            while (t1[i] > sstar) and (m[i]<k-1) :
                m[i] = m[i] + 1
                t1[i] = min(t1[i], cmidd(X.iloc[:,i], # feature i
                                         y,  # target
                                         X.iloc[:, int(F[int(m[i])])] # conditionned on selected features
                                        )
                           )
            if t1[i] > sstar:
                sstar = t1[i]
                F[k+1] = i
                
F = np.array(F[F>-100])
F = F.astype(int)
t1 = t1[F]

n_estimators = 100
model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
model.fit(X_train_data.iloc[:,F[:10]],y_train)
y_pred = model.predict(X_test_data.iloc[:,F[:10]])
    
print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)


#####################MIM-SVR

start = time.time()    

F, J_CMI, MIfy = LCSI.lcsi(X_train_data, y_train, beta=0, gamma=0)

n_estimators = 1000
regr = make_pipeline(StandardScaler(),  LinearSVR(random_state=0, tol=1e-3))
regr.fit(X_train_data.iloc[:,F[:10]], y_train)

y_pred = regr.predict(X_test_data.iloc[:,F[:10]])


print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)

#######################MIM-KNNr
      
start = time.time()    
    
F, J_CMI, MIfy = LCSI.lcsi(X_train_data, y_train, beta=0, gamma=0)

error_rate = []
for i in range(1,25):
 knn = KNeighborsRegressor(n_neighbors=i)
 knn.fit(X_train_data.iloc[:,F[:10]],y_train)
 pred_i = knn.predict(X_test_data.iloc[:,F[:10]])
 error_rate.append(np.mean(pred_i != y_test))

knn = KNeighborsRegressor(n_neighbors= error_rate.index(min(error_rate)) if error_rate.index(min(error_rate)) > 0 else 1)
knn.fit(X_train_data.iloc[:,F[:10]],y_train)
y_pred = knn.predict(X_test_data.iloc[:,F[:10]])


print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)


############################### MIM-RFR

start = time.time()    
    
F, J_CMI, MIfy = LCSI.lcsi(X_train_data, y_train, beta=0, gamma=0)

n_estimators = 1000
model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
model.fit(X_train_data.iloc[:,F[:10]],y_train)
y_pred = model.predict(X_test_data.iloc[:,F[:10]])


print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)


###################################


start = time.time()    
        
regr = AdaBoostRegressor(random_state=0, n_estimators=100)
regr.fit(X_train_data, y_train)

imp = regr.feature_importances_

n_estimators = 1000
regr = make_pipeline(StandardScaler(),  LinearSVR(random_state=0, tol=1e-3))
regr.fit(X_train_data[X_train_data.columns[imp > 0]], y_train)

y_pred = regr.predict(X_test_data[X_train_data.columns[imp > 0]])
      


print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)


#######################################

start = time.time()    
    
regr = AdaBoostRegressor(random_state=0, n_estimators=100)
regr.fit(X_train_data, y_train)

imp = regr.feature_importances_
error_rate = []
for i in range(1,25):
 knn = KNeighborsRegressor(n_neighbors=i)
 knn.fit(X_train_data[X_train_data.columns[imp > 0]],y_train)
 pred_i = knn.predict(X_test_data[X_test_data.columns[imp > 0]])
 error_rate.append(np.mean(pred_i != y_test))

knn = KNeighborsRegressor(n_neighbors= error_rate.index(min(error_rate)) if error_rate.index(min(error_rate)) > 0 else 1)
knn.fit(X_train_data[X_train_data.columns[imp > 0]],y_train)
y_pred = knn.predict(X_test_data[X_test_data.columns[imp > 0]])



print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)


#######################################Adaboost+RFR

start = time.time()    
       
regr = AdaBoostRegressor(random_state=0, n_estimators=100)
regr.fit(X_train_data, y_train)

imp = regr.feature_importances_
n_estimators = 100
model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
model.fit(X_train_data[X_train_data.columns[imp > 0]],y_train)
y_pred = model.predict(X_test_data[X_test_data.columns[imp > 0]])


print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)


################################

start = time.time()    
        
config = {'algorithm': 'C4.5'}
df = X_train_data
df["Decision"] = y_train
#########
model = chef.fit(df, config)
y_pred = []
for index, instance in X_test_data.iterrows():
    y_pred.append(chef.predict(model,instance))



print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)

####################################

import time
start = time.time()
ls = ["Polyuria" , "Polydipsia" ,"delayed healing" , "muscle stiffness"]
X_train = X_train_data[ls]
X_test = X_test_data[ls]
n_estimators =1000
model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


print(sum((y_pred-y_test)**2))
print(mean_squared_error(y_pred,y_test))
print(time.time()-start)
