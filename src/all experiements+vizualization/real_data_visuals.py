# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:57:47 2020

@author: KIIT
"""

import pandas as pd
import os 
import seaborn as sns

df = pd.read_csv(r'C:\Users\KIIT\Desktop\Causal\data\diabetes1.csv')
from sklearn. preprocessing import LabelEncoder
df.iloc[:,1:] = df.iloc[:,1:].apply(LabelEncoder().fit_transform)

df1 = df[df.Age <50]
df2 = df[df.Age >= 50]

import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from skfeature.function.information_theoretical_based import LCSI
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR
mse_arr=[]
sse_arr=[]

for i in range(8):
    temp_sse=[]
    temp_mse=[]
    temp = df1.sample(100)
    X_train_data, y_train_data = temp.drop(['class'], axis = 1),temp.iloc[:,16]
    X_test_data, y_test_data = df2.drop(['class'], axis = 1),df2.iloc[:,16]

    F, J_CMI, MIfy = LCSI.lcsi(X_train_data, y_train_data, beta=0, gamma=0)
    
    n_estimators = 1000
    regr = make_pipeline(StandardScaler(),  LinearSVR(random_state=0, tol=1e-3))
    regr.fit(X_train_data.iloc[:,F[:10]], y_train_data)
    
    y_pred = regr.predict(X_test_data.iloc[:,F[:10]])
      
    temp_sse.append(sum((y_pred- df2['class'])**2))
    temp_mse.append(mean_squared_error(y_pred,df2['class']))
    
    regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    regr.fit(X_train_data, y_train_data)
    
    imp = regr.feature_importances_
    regr = make_pipeline(StandardScaler(),  LinearSVR(random_state=0, tol=1e-3))
    regr.fit(X_train_data[X_train_data.columns[imp > 0]], y_train_data)
    
    y_pred = regr.predict(X_test_data[X_train_data.columns[imp > 0]])
    temp_sse.append(sum((y_pred- df2['class'])**2))
    temp_mse.append(mean_squared_error(y_pred,df2['class']))
    
    
    model = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)
    model.fit(X_train_data,y_train_data)
    y_pred = model.predict(df2.drop(['class'], axis=1))
    temp_sse.append(sum((y_pred- df2['class'])**2))
    temp_mse.append(mean_squared_error(y_pred,df2['class']))
    
    feat= ['Polyuria', 'Alopecia','visual blurring','delayed healing', 'muscle stiffness']
    model = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)
    model.fit(X_train_data,y_train_data)
    y_pred = model.predict(df2.drop(['class'], axis=1))
    temp_sse.append(sum((y_pred- df2['class'])**2))
    temp_mse.append(mean_squared_error(y_pred,df2['class']))
    
    mse_arr.append(temp_mse)
    sse_arr.append(temp_sse)

import numpy as np
mse_arr = np.array(mse_arr)    
sse_arr = np.array(sse_arr)    


titles = 'AGE change'
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 5)) 

labels = ['Baseline','CMIM+SVR','Adaboost+RFR','RTCL'] 
bplot = ax.boxplot(mse_arr,vert=True,patch_artist=True,  labels =labels) 
colors = ['pink', 'lightblue', 'lightgreen','blue']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
ax.set_title(titles)
ax.set_xlabel('Methodologies')
ax.set_ylabel('SSE')

plt.xticks(rotation = 20) 


mse_df = pd.DataFrame(np.array(mse_arr))
mse_df.to_csv('{0}_mse.csv'.format(titles), index=False, header=False)

sse_df = pd.DataFrame(np.array(sse_arr))
sse_df.to_csv('{0}_sse.csv'.format(titles), index=False, header=False)
