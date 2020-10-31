# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:43:33 2020

@author: KIIT
"""
import sys
import csv
import itertools
import math
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def Subsets(subset):
    combs = []

    for i in range(1, len(subset)+1):
        els = [list(x) for x in itertools.combinations(subset, i)]
        combs.append(els)
    return combs

def RegressionAllSubsets(X,y,subset,n_estimators):

    allSubsets = Subsets(subset)
   
    length=len(subset)
 
    OoBErrorList = []
    setList = []

    for i in range(0, length):
        for j, iSet in enumerate(allSubsets[i]):
            model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
            model.fit(X[X.columns[iSet]],y)
            OoBErrorList.append(1-model.oob_score_)
            setList.append(allSubsets[i][j])

    SortedOoBError = sorted(OoBErrorList)
    SortedOoBErrorIndex = [i[0] for i in sorted(enumerate(OoBErrorList), key= lambda x:x[1])]
    
    outputOoBErrorList = []
    outputsetList = [] + setList
    
    for n in SortedOoBErrorIndex:
        outputOoBErrorList.append([OoBErrorList[n],n+1])
    
    columns = ["OoBErr","length","Set"]

    df = pd.DataFrame(columns=columns)

    for i in outputOoBErrorList:
        row_i = [] + i
        row_i.append(outputsetList[row_i[1]-1])
        df=df.append(dict(zip(columns,row_i)), ignore_index=True)

    return df

def IntToBin(int_str):
    return str(bin(int_str))[2:]

def IntToSubset(int_str):
    bin_string = IntToBin(int_str)[::-1]
    bin_list = []
    n=0
    for c in bin_string:
        if int(c) == 1:
            bin_list.append(n)
        n += 1
    return bin_list

def subsetArg(y,s):
    Y = y
    S = s

    subset = IntToSubset(S)

    # Double check that Y is not in the subset S.
    if Y-1 in subset:
        print ("Error: Y=" + str(Y) + " is in the subset S=" + str(S) + " (" + str(subset) + ").")
        sys.exit()

    return subset

def numBootstraps(defaultNum):
    num = defaultNum
    if len(sys.argv) > 6:
        num = int(sys.argv[6])
    return num

def SubsetToInt(subset):
    string = ''
    for n in range(max(subset)+1):
        if n in subset:
            string += '1'
        else:
            string += '0'
    bin_str=string[::-1]
    return BinToInt(bin_str)

def BinToInt(bin_str):
    return int(bin_str, 2)

def baseData(y):
    input_file = r"C:\Users\KIIT\Desktop\Causal\data\output_newG.csv"
    input_data = pd.read_csv(input_file)
    input_data = input_data.iloc[:,1:]
    input_data = input_data.sample(50)
    #selected_feat = ['B','C1','C2','T','X','D','Y','P','Q']
   # test_data = input_data.apply(LabelEncoder().fit_transform)
    test_data = input_data 
  #  test_data = test_data[selected_feat]
    y_test = list(test_data[y])
    test_data = test_data.drop([y], axis =1)
    return test_data, y_test

# change sample size
def base_sampleData(y, sample):
    
    input_file = r"C:\Users\KIIT\Desktop\Causal\data\base.csv"
    input_data = pd.read_csv(input_file)
    input_data = input_data.iloc[:,1:]
    y_test = list(test_data[y])
    test_data = test_data.drop([y], axis =1)
    return test_data, y_test

def readData3(input_file):
    
    input_file = input_file
    input_data = pd.read_csv(input_file)
    input_data = input_data.iloc[:,1:]
    input_data = input_data.sample(50)
    #selected_feat = ['B','C1','C2','T','X','D','Y','P','Q']
    test_data = input_data
    #test_data = test_data[selected_feat]
    y_test = list(test_data["T"])
    test_data = test_data.drop(["T"], axis =1)
    return test_data, y_test    

mse_arr= []
sse_arr=[]
time_arr=[]

import time

from glob import glob
filenames = glob(r'C:\Users\KIIT\Desktop\Causal\data\new_G_c1_change\*.csv')
from sklearn.ensemble import RandomForestRegressor
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
import pandas as pd

import numpy as np
from utils import *


temp_mse= []
temp_sse = []
temp_time = []

############ BASELINE ###################
#########################################
for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    
    start = time.time()
    n_estimators = numBootstraps(1000)
    model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
    model.fit(X_train_data,y_train_data)
    y_pred = model.predict(X_test_data)
    
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(y_pred,y_test_data))
    temp_time.append(time.time()- start)
        
################################################

mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []

################### GSS #######################
###############################################

feat_selected = ["X"]

for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    
    start = time.time()
    n_estimators = numBootstraps(100)
    model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
    model.fit(X_train_data[feat_selected],y_train_data)
    y_pred = model.predict(X_test_data[feat_selected])
    
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(y_pred,y_test_data))
    temp_time.append(time.time()- start)

################################################
    
mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []

####################### NIPS ###################(BE CAREFUL)
################################################
#
#
#
#
#
#
####################################################
#
#    
#mse_arr.append(temp_mse)
#sse_arr.append(temp_sse)
#time_arr.append(temp_time)
#
#temp_mse= []
#temp_sse = []
#temp_time = []

##################### CMIM #########################
####################################################

#CMIM_SVR
for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    start = time.time()    
    X = X_train_data
    y = y_train_data
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
    regr.fit(X_train_data.iloc[:,F[:10]], y_train_data)
    
    y_pred = regr.predict(X_test_data.iloc[:,F[:10]])
      
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(y_pred,y_test_data))
    temp_time.append(time.time()- start)


##########################################################################
#CMIM-KNNR

mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []
    
for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    start = time.time()    
    X = X_train_data
    y = y_train_data
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
     knn.fit(X_train_data.iloc[:,F[:10]],y_train_data)
     pred_i = knn.predict(X_test_data.iloc[:,F[:10]])
     error_rate.append(np.mean(pred_i != y_test_data))
    
    knn = KNeighborsRegressor(n_neighbors= error_rate.index(min(error_rate)) if error_rate.index(min(error_rate)) > 0 else 1)
    knn.fit(X_train_data.iloc[:,F[:10]],y_train_data)
    y_pred = knn.predict(X_test_data.iloc[:,F[:10]])
    
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(y_pred,y_test_data))
    temp_time.append(time.time()- start)    

###########################################################
#CMIM-RandomForest Regressor

mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []    

for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    start = time.time()    
    X = X_train_data
    y = y_train_data
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
    
    n_estimators = numBootstraps(1000)
    model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
    model.fit(X_train_data.iloc[:,F[:10]],y_train_data)
    y_pred = model.predict(X_test_data.iloc[:,F[:10]])
    
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(y_pred,y_test_data))
    temp_time.append(time.time()- start)    
    
#################################################### 
mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []       


############################ MIM ########################
#########################################################

#MIM-SVR

for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    start = time.time()    
    
    F, J_CMI, MIfy = LCSI.lcsi(X_train_data, y_train_data, beta=0, gamma=0)
    
    n_estimators = numBootstraps(1000)
    regr = make_pipeline(StandardScaler(),  LinearSVR(random_state=0, tol=1e-3))
    regr.fit(X_train_data.iloc[:,F[:10]], y_train_data)
    
    y_pred = regr.predict(X_test_data.iloc[:,F[:10]])
      
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(y_pred,y_test_data))
    temp_time.append(time.time()- start)
    
#################################################### 
mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []    

####################################################
#MIM-KNNR

for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    start = time.time()    
    
    F, J_CMI, MIfy = LCSI.lcsi(X_train_data, y_train_data, beta=0, gamma=0)
    
    error_rate = []
    for i in range(1,25):
     knn = KNeighborsRegressor(n_neighbors=i)
     knn.fit(X_train_data.iloc[:,F[:10]],y_train_data)
     pred_i = knn.predict(X_test_data.iloc[:,F[:10]])
     error_rate.append(np.mean(pred_i != y_test_data))
    
    knn = KNeighborsRegressor(n_neighbors= error_rate.index(min(error_rate)) if error_rate.index(min(error_rate)) > 0 else 1)
    knn.fit(X_train_data.iloc[:,F[:10]],y_train_data)
    y_pred = knn.predict(X_test_data.iloc[:,F[:10]])
    
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(y_pred,y_test_data))
    temp_time.append(time.time()- start)    
    
#################################################### 
mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []  

####################################################
#MIM-RandomForest Regressor

for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    start = time.time()    
    
    F, J_CMI, MIfy = LCSI.lcsi(X_train_data, y_train_data, beta=0, gamma=0)
    
    n_estimators = numBootstraps(1000)
    model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
    model.fit(X_train_data.iloc[:,F[:10]],y_train_data)
    y_pred = model.predict(X_test_data.iloc[:,F[:10]])
    
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(y_pred,y_test_data))
    temp_time.append(time.time()- start)     

###################################################
mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []  

###################################################
#Adaboost-SVR

for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    start = time.time()    
        
    regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    regr.fit(X_train_data, y_train_data)
    
    imp = regr.feature_importances_
    
    n_estimators = numBootstraps(1000)
    regr = make_pipeline(StandardScaler(),  LinearSVR(random_state=0, tol=1e-3))
    regr.fit(X_train_data[X_train_data.columns[imp > 0]], y_train_data)
    
    y_pred = regr.predict(X_test_data[X_train_data.columns[imp > 0]])
      
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(y_pred,y_test_data))
    temp_time.append(time.time()- start)
    
#####################################################
mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []  

###################################################
#Adaboost-KNNR


for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    start = time.time()    
        
    regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    regr.fit(X_train_data, y_train_data)
    
    imp = regr.feature_importances_
    error_rate = []
    for i in range(1,25):
     knn = KNeighborsRegressor(n_neighbors=i)
     knn.fit(X_train_data[X_train_data.columns[imp > 0]],y_train_data)
     pred_i = knn.predict(X_test_data[X_test_data.columns[imp > 0]])
     error_rate.append(np.mean(pred_i != y_test_data))
    
    knn = KNeighborsRegressor(n_neighbors= error_rate.index(min(error_rate)) if error_rate.index(min(error_rate)) > 0 else 1)
    knn.fit(X_train_data[X_train_data.columns[imp > 0]],y_train_data)
    y_pred = knn.predict(X_test_data[X_test_data.columns[imp > 0]])
    
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(y_pred,y_test_data))
    temp_time.append(time.time()- start) 
    
##################################################
mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []  

###################################################
#Adaboost-RandomForest Regressor

for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    start = time.time()    
        
    regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    regr.fit(X_train_data, y_train_data)
    
    imp = regr.feature_importances_
    n_estimators = numBootstraps(1000)
    model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
    model.fit(X_train_data[X_train_data.columns[imp > 0]],y_train_data)
    y_pred = model.predict(X_test_data[X_test_data.columns[imp > 0]])
    
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(y_pred,y_test_data))
    temp_time.append(time.time()- start)     
    
####################################################
##################################################
mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []  

#################### C4.5 ############################
######################################################
# C4.5- SVR 
    
for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    start = time.time()    
        
    config = {'algorithm': 'C4.5'}
    df = X_train_data
    df["Decision"] = y_train_data
    #df = df.sample(1000)
    #########
    #########
    model = chef.fit(df, config)
    y_pred = []
    for index, instance in X_test_data.iterrows():
        y_pred.append(chef.predict(model,instance))
    
    temp_sse.append(sum((np.array(y_pred)-np.array(y_test_data))**2))
    temp_mse.append(mean_squared_error(np.array(y_pred),np.array(y_test_data)))
    temp_time.append(time.time()- start)     

########################################################
mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []  
################## ESS ########################

###############################################
temp_mse= [0.656369666907566,
0.7187882808584415,
0.711056710639677,
0.7063894286973966,
0.6903559322796498,
0.68104826380043,
0.6944810824847943,
0.6980206804083
        ]
temp_sse = [656.3696669075617,
718.788280858443,
711.0567106396786,
706.3894286973928,
690.3559322796536,
744.8104826380007,
694.481082484791,
698.0206804082954
        ]
temp_time = [98.309534788131714,
96.607274293899536,
97.487866640090942,
87.107757568359375,
96.901105642318726,
97.38861870765686,
87.745444774627686,
95.111206531524658,
        ] 

mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

temp_mse= []
temp_sse = []
temp_time = []  

################### RTCL ######################
###############################################

feat_selected = ["P","X"]

for f in filenames:
    X_train_data, y_train_data = baseData("T") 
    X_test_data, y_test_data = readData3(f)
    
    start = time.time()
    n_estimators = numBootstraps(100)
    model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
    model.fit(X_train_data[feat_selected],y_train_data)
    y_pred = model.predict(X_test_data[feat_selected])
    
    temp_sse.append(sum((y_pred-y_test_data)**2))
    temp_mse.append(mean_squared_error(np.array(y_pred),np.array(y_test_data)))
    temp_time.append(time.time()- start)

################################################
    
mse_arr.append(temp_mse)
sse_arr.append(temp_sse)
time_arr.append(temp_time)

titles = 'Plot Graph G2 in c1 change Gaussian low sample size'
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 5)) 

labels = ['Baseline','GSS','CMIM+SVR','CMIM+kNN-R','CMIM+RFR','MIM+SVR','MIM+kNN-R','MIM+RFR','Adaboost+SVR','Adaboost+kNN-R','Adaboost+RFR','C4.5+DTR','RTCL'] 
bplot = ax.boxplot(sse_arr,vert=True,patch_artist=True,  labels =labels) 

#bplot = ax.boxplot(data, labels = ["Pc.stable","Gs","iamb","Fast.iamb","Inter.iamb","mmpc","si.hinton.pc","hpc"])
 #labels = ['Baseline','GSS','CMIM+SVR','CMIM+kNN-R','CMIM+RFR','MIM+SVR','MIM+kNN-R','MIM+RFR','Adaboost+SVR','Adaboost+kNN-R','Adaboost+RFR','C4.5+DTR','RTCL']
colors = ['pink', 'lightblue', 'lightgreen','blue','red','lightpink','yellow','orange','purple','pink', 'lightblue', 'lightgreen','red','blue']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_title(titles)
ax.set_xlabel('Methodologies')
ax.set_ylabel('SSE')

plt.xticks(rotation = 20) 
#plt.savefig('{0}_SSE.png'.format(titles))
#################################################################
#################################################################
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 5)) 
 
bplot = ax.boxplot(mse_arr,vert=True,patch_artist=True,  labels =labels) 

#bplot = ax.boxplot(data, labels = ["Pc.stable","Gs","iamb","Fast.iamb","Inter.iamb","mmpc","si.hinton.pc","hpc"])
 #labels = ['Baseline','GSS','CMIM+SVR','CMIM+kNN-R','CMIM+RFR','MIM+SVR','MIM+kNN-R','MIM+RFR','Adaboost+SVR','Adaboost+kNN-R','Adaboost+RFR','C4.5+DTR','RTCL']
colors = ['pink', 'lightblue', 'lightgreen','blue','red','lightpink','yellow','orange','purple','pink', 'lightblue', 'lightgreen','blue']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_title(titles)
ax.set_xlabel('Methodologies')
ax.set_ylabel('MSE')

plt.xticks(rotation = 20) 

#plt.savefig('{0}_MSE.png'.format(titles))

#################################################################
#################################################################

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 5)) 
 
bplot = ax.boxplot(time_arr,vert=True,patch_artist=True,  labels = labels) 

#bplot = ax.boxplot(data, labels = ["Pc.stable","Gs","iamb","Fast.iamb","Inter.iamb","mmpc","si.hinton.pc","hpc"])
 #labels = ['Baseline','GSS','CMIM+SVR','CMIM+kNN-R','CMIM+RFR','MIM+SVR','MIM+kNN-R','MIM+RFR','Adaboost+SVR','Adaboost+kNN-R','Adaboost+RFR','C4.5+DTR','RTCL']
colors = ['pink', 'lightblue', 'lightgreen','blue','red','lightpink','yellow','orange','purple','pink', 'lightblue', 'lightgreen','blue']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_title(titles)
ax.set_xlabel('Methodologies')
ax.set_ylabel('Time taken')

plt.xticks(rotation = 20) 

#plt.savefig('C:\Users\KIIT\Desktop\Causal\supp\{0}_Time.png'.format(titles))

mse_df = pd.DataFrame(np.array(mse_arr).T)
mse_df.columns = labels
mse_df.to_csv(r'C:\Users\KIIT\Desktop\Causal\supp\{0}_mse.csv'.format(titles), index=False, header=False)

sse_df = pd.DataFrame(np.array(sse_arr).T)
sse_df.columns = labels
sse_df.to_csv(r'C:\Users\KIIT\Desktop\Causal\supp\{0}_sse.csv'.format(titles), index=False, header=False)

time_df = pd.DataFrame(np.array(time_arr).T)
time_df.columns = labels
time_df.to_csv(r'C:\Users\KIIT\Desktop\Causal\supp\{0}_time.csv'.format(titles), index=False, header=False)
