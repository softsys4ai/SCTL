from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

import numpy as np
from utils import *

def fast_cmim(X, y, **kwargs):
    """
    This function implements the CMIM feature selection.
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete numpy array
    y: {numpy array}, shape (n_samples,)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    t1: {numpy array}, shape: (n_features,)
        minimal corresponding mutual information between selected features and response when 
        conditionned on a previously selected feature
    Reference
    ---------
    Fleuret 2004 - Fast Binary Feature Selection with Conditional Mutual Information
    http://www.idiap.ch/~fleuret/papers/fleuret-jmlr2004.pdf
    """

    n_samples, n_features = X.shape
    is_n_selected_features_specified = False

    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True
        F = np.nan * np.zeros(n_selected_features)
    else:
        F = np.nan * np.zeros(n_features)

    # t1
    t1 = np.zeros(n_features)
    
    # m is a counting indicator
    m = np.zeros(n_features) - 1
    
    for i in range(n_features):
        f = X[:, i]
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
            f_select = X[:, idx]

        if is_n_selected_features_specified:
            if np.sum(~np.isnan(F)) == n_selected_features:
                break

        sstar = -1000000 # start with really low value for best partial score sstar 
        for i in range(n_features):
            
            if i not in F:
                
                while (t1[i] > sstar) and (m[i]<k-1) :
                    m[i] = m[i] + 1
                    t1[i] = min(t1[i], cmidd(X[:,i], # feature i
                                             y,  # target
                                             X[:, int(F[int(m[i])])] # conditionned on selected features
                                            )
                               )
                if t1[i] > sstar:
                    sstar = t1[i]
                    F[k+1] = i
                    
    F = np.array(F[F>-100])
    F = F.astype(int)
    t1 = t1[F]
return (F, t1)

#CMIM-SVR
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

regr = make_pipeline(StandardScaler(),  LinearSVR(random_state=0, tol=1e-3))

X_train_data = X.iloc[:,F[:10]]
regr.fit(X_train_data, y_train_data)
y_pred = regr.predict(X_test_data.iloc[:,F[:10]])

print(sum((y_pred-y_test_data)**2))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_pred,y_test_data))

error_rate = []
for i in range(1,40):
 knn = KNeighborsRegressor(n_neighbors=i)
 knn.fit(X_train_data,y_train_data)
 pred_i = knn.predict(X_test_data.iloc[:,F[:10]])
 error_rate.append(np.mean(pred_i != y_test_data))

print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
knn = KNeighborsRegressor(n_neighbors= error_rate.index(min(error_rate)) if error_rate.index(min(error_rate)) > 0 else 1)
knn.fit(X_train_data,y_train_data)
y_pred = knn.predict(X_test_data.iloc[:,F[:10]])

print(sum((y_pred-y_test_data)**2))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_pred,y_test_data))


from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

classifiers = [('SVR', LinearSVR(random_state=0, tol=1e-3)),
              ('RandomForestRegressor', RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1))]
              
n_features = X_train_data.shape[1]
npieces = get_i(n_features)


"""
FCBF
"""
fcbf = FCBF()
t0 = time.time()
fcbf.fit(X_train_data, X_test_data)
elapsed_t = time.time()-t0

k = len(fcbf.idx_sel) #Number of selected features for FCBFK and FCBFiP

"""
FCBF#
"""
fcbfk = FCBFK(k = k)
t0 = time.time()
fcbfk.fit(X_train_data, X_test_data)
elapsed_t = time.time()-t0
    
from skfeature.utility.mutual_information import su_calculation
#
#
#def fcbf(X, y, **kwargs):
#    """
#    This function implements Fast Correlation Based Filter algorithm
#    Input
#    -----
#    X: {numpy array}, shape (n_samples, n_features)
#        input data, guaranteed to be discrete
#    y: {numpy array}, shape (n_samples,)
#        input class labels
#    kwargs: {dictionary}
#        delta: {float}
#            delta is a threshold parameter, the default value of delta is 0
#    Output
#    ------
#    F: {numpy array}, shape (n_features,)
#        index of selected features, F[0] is the most important feature
#    SU: {numpy array}, shape (n_features,)
#        symmetrical uncertainty of selected features
#    Reference
#    ---------
#        Yu, Lei and Liu, Huan. "Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution." ICML 2003.
#    """
#
#    n_samples, n_features = X.shape
#    if 'delta' in kwargs.keys():
#        delta = kwargs['delta']
#    else:
#        # the default value of delta is 0
#        delta = 0
#
#    # t1[:,0] stores index of features, t1[:,1] stores symmetrical uncertainty of features
#    t1 = np.zeros((n_features, 2), dtype='object')
#    for i in range(n_features):
#        f = X.iloc[:, i]
#        t1[i, 0] = i
#        t1[i, 1] = su_calculation(f, y)
#    s_list = t1[t1[:, 1] > delta, :]
#    # index of selected features, initialized to be empty
#    F = []
#    # Symmetrical uncertainty of selected features
#    SU = []
#    while len(s_list) != 0:
#        # select the largest su inside s_list
#        idx = np.argmax(s_list[:, 1])
#        # record the index of the feature with the largest su
#        fp = X.iloc[:, s_list[idx, 0]]
#        np.delete(s_list, idx, 0)
#        F.append(s_list[idx, 0])
#        SU.append(s_list[idx, 1])
#        for i in s_list[:, 0]:
#            fi = X.iloc[:, i]
#            if su_calculation(fp, fi) >= t1[i, 1]:
#                # construct the mask for feature whose su is larger than su(fp,y)
#                idx = s_list[:, 0] != i
#                idx = np.array([idx, idx])
#                idx = np.transpose(idx)
#                # delete the feature by using the mask
#                s_list = s_list[idx]
#                length = len(s_list)//2
#                s_list = s_list.reshape((length, 2))
#    return np.array(F, dtype=int), np.array(SU)
#
#feat_index, sym_arr = fcbf(X_train_data.iloc[:,:5], X_test_data.iloc[:,:5])


#MIM
from skfeature.function.information_theoretical_based import LCSI
F, J_CMI, MIfy = LCSI.lcsi(X_train_data, y_train_data, beta=0, gamma=0)

from sklearn.ensemble import AdaBoostRegressor
regr = AdaBoostRegressor(random_state=0, n_estimators=100)
regr.fit(X_train_data, y_train_data)

imp = regr.feature_importances_

X_train_data.columns[imp > 0]


from chefboost import Chefboost as chef
import pandas as pd

config = {'algorithm': 'C4.5'}
df = X_train_data
df["Decision"] = y_train_data
model = chef.fit(df, config)

prediction = chef.predict(X_test_data)