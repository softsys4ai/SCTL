# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:34:24 2020

@author: KIIT
"""
import utils
import numpy as np
import scipy as sp
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
import itertools
import utils
import sys

import numpy as np
import sys
sys.path.append('code')
import subset_search
from sklearn import linear_model


np.random.seed(12)

n_examples_task = 1000
n_tasks = 2
n_test_tasks = 100
n_predictors = 18
n_ex = []

#########################


alpha = np.random.uniform(-1, 2.5, 18)
sigma = 1.5
sx1 = 1
sx2 = 0.1
sx3 = 1

train_x = np.zeros((1, n_predictors))
train_y = np.zeros(1)

use_hsic = True
return_mse = False
delta = 0.05

###############################

for task in range(n_tasks):
    u =  np.random.uniform(-1, 1)
    c1 = np.random.normal(0, sx1,(n_examples_task, 1))
    c2 = np.random.normal(0, sx3, (n_examples_task,1))
    x = c2 + np.random.normal(0, sx2, (n_examples_task, 1))
    t = x + np.random.normal(0, sx1, (n_examples_task, 1))
    y = alpha[0] *c1 + alpha[1] * t + np.random.normal(0, sigma, (n_examples_task, 1))
    d = np.random.normal(0, 2.5, (n_examples_task, 1))
    b = c2 + d + np.random.normal(0, 4, (n_examples_task, 1))
    p = t + np.random.normal(0, sx2, (n_examples_task, 1))+ b
    q = p + np.random.normal(0, sx2, (n_examples_task, 1))
    k = np.random.normal(0, 4.5, (n_examples_task, 1))
    l = k + np.random.normal(0, 3.5, (n_examples_task, 1))
    m = k+l+ np.random.normal(0, 6.5, (n_examples_task, 1))
    j = k + np.random.normal(0, 2.5, (n_examples_task, 1))
    n = j+m+ np.random.normal(0, 5.5, (n_examples_task, 1))
    f = np.random.normal(0, 5.5, (n_examples_task, 1))
    g = np.random.normal(0, 5, (n_examples_task, 1))
    h = np.random.normal(0, 2.8, (n_examples_task, 1))
    e = g+h+ np.random.normal(0, 4.5, (n_examples_task, 1))
    i = e+ np.random.normal(0, 5.5, (n_examples_task, 1))
    x_task = np.concatenate([c1,c2,x,y,e,q,d,b,k,l,n,j,t,f,g,h,m,i],axis = 1)
    train_x = np.append(train_x, x_task, axis = 0)
    train_y = np.append(train_y, p)
    n_ex.append(n_examples_task)

n_ex = np.array(n_ex)
train_x =  train_x[1:, :]
train_y = train_y[1:, np.newaxis]

test_x = np.zeros((1, n_predictors))
test_y = np.zeros(1)

###################################

for task in range(n_test_tasks):

    gamma_task = np.random.uniform(-1,1)
    x1 = np.random.normal(0,sx1,(n_examples_task,1))
    x3 = np.random.normal(0,sx3,(n_examples_task,1))
    y = alpha[0]*x1 + alpha[1]*x3 + np.random.normal(
      0,sigma,(n_examples_task,1))
    x2 = gamma_task*y + np.random.normal(0,sx2,(n_examples_task,1))

    x_task = np.concatenate([x1, x2, x3],axis = 1)
    test_x = np.append(test_x, x_task, axis = 0)
    test_y = np.append(test_y, y)

test_x = test_x[1:,:]
test_y = test_y[1:,np.newaxis]

####################################

import time
start = time.time()
valid_split= 0.1
x = train_x
y = train_y      
n_ex_cum = np.append(0, np.cumsum(n_ex))
n_ex_train, n_ex_valid = [], []
train_x, train_y, valid_x, valid_y = [], [], [], []

for i in range(len(n_ex)):
    n_train_task = int((1 - valid_split) * n_ex[i])
    train_x.append(x[n_ex_cum[i]:n_ex_cum[i] + n_train_task])
    train_y.append(y[n_ex_cum[i]:n_ex_cum[i] + n_train_task])

    valid_x.append(x[n_ex_cum[i] + n_train_task:n_ex_cum[i + 1]])
    valid_y.append(y[n_ex_cum[i] + n_train_task:n_ex_cum[i + 1]])
    
    n_ex_train.append(n_train_task)
    n_ex_valid.append(n_ex[i] - n_train_task)

train_x = np.concatenate(train_x, 0)
valid_x = np.concatenate(valid_x, 0)
train_y = np.concatenate(train_y, 0)
valid_y = np.concatenate(valid_y, 0)

n_ex_train = np.array(n_ex_train)
n_ex_valid = np.array(n_ex_valid)
    
#########################
num_tasks = len(n_ex)
n_ex_cum = np.cumsum(n_ex)

index_task = 0
best_subset = []
accepted_sets = []
accepted_mse = []
all_sets = []
all_pvals = []

num_s = np.sum(n_ex)
num_s_valid = np.sum(n_ex_valid)
best_mse = 1e10

rang = np.arange(train_x.shape[1])
maxevT = -10
maxpval = 0
num_accepted = 0
current_inter = np.arange(train_x.shape[1])
#############################################

pred_valid = np.mean(train_y)
residual = valid_y - pred_valid

if use_hsic:
    X,nEx = valid_y, n_ex_valid
    nExCum = np.cumsum(nEx)
    domains = np.zeros((np.sum(nEx),np.sum(nEx)))
    currentIndex = 0
    
    for i in range(len(nEx)):
    	domains[currentIndex:nExCum[i], currentIndex:nExCum[i]] = np.ones((nEx[i], nEx[i]))
    	currentIndex = nExCum[i]
    
    valid_dom = domains
    ls = utils.np_getDistances(residual, residual)
    sx = 0.5 * np.median(ls.flatten())
    X,Y, sigmaX, sigmaY, DomKer = residual, valid_dom, sx, 1, valid_dom
    
    n = X.T.shape[1]
    X,sX = X,sigmaX
    
    KernelX = (X[:,:, np.newaxis] - X.T).T
    KernelX = np.exp( -1./(2*sX) * np.linalg.norm(KernelX, axis=1))
    
    KernelY = DomKer
    
    coef = 1./n
    s1 = np.matmul(KernelX,KernelY)
    HSIC = (coef**2 * (np.sum(KernelX*KernelY))) + (coef**4 * (np.sum(KernelX) * np.sum(KernelY))) - (2 * coef**3 * np.sum(np.sum(KernelX,axis=1) * np.sum(KernelY, axis=1)))
    	 
    	#Get sums of Kernels
    KXsum = np.sum(KernelX)
    KYsum = np.sum(KernelY)
    
    xMu = 1./(n*(n-1))*(KXsum - n)
    yMu = 1./(n*(n-1))*(KYsum - n)
    V1 = coef**2*np.sum(KernelX*KernelX) + coef**4*KXsum**2 - 2*coef**3*np.sum(np.sum(KernelX,axis=1)**2)
    V2 = coef**2*np.sum(KernelY*KernelY) + coef**4*KYsum**2 - 2*coef**3*np.sum(np.sum(KernelY,axis=1)**2)
    meanH0 = (1. + xMu*yMu - xMu - yMu)/n
    varH0 = 2.*(n-4)*(n-5)/(n*(n-1.)*(n-2.)*(n-3.))*V1*V2
    
    	#Parameters of the Gamma
    a = meanH0**2/varH0
    b = n * varH0/meanH0
    
    stat, a, b = n*HSIC, a, b
    pvals = 1. - sp.stats.gamma.cdf(stat, a, scale=b)
#else:
#    residTup = utils.levene_pval(residual, n_ex, num_tasks)
#    pvals = sp.stats.levene(*residTup)[1]

if (pvals > alpha[0]):
    mse_current  = np.mean((valid_y - pred_valid) ** 2)
    if mse_current < best_mse:
        best_mse = mse_current
        best_subset = []
        accepted_sets.append([])
        accepted_mse.append(mse_current)

all_sets.append([])
all_pvals.append(pvals)


for i in range(1, rang.size + 1):
    for s in itertools.combinations(rang, i):
        currentIndex = rang[np.array(s)]
        regr = linear_model.LinearRegression()
        
        #Train regression with given subset on training data
        regr.fit(train_x[:, currentIndex], 
                 train_y.flatten())

        #Compute mse for the validation set
        pred = regr.predict(
          valid_x[:, currentIndex])[:,np.newaxis]

        #Compute residual
        residual = valid_y - pred

        if use_hsic:
            valid_dom = utils.mat_hsic(valid_y, n_ex_valid)
            ls = utils.np_getDistances(residual, residual)
            sx= 0.5 * np.median(ls.flatten())
            stat, a, b = utils.numpy_HsicGammaTest(
                residual, valid_dom, sx, 1, DomKer = valid_dom)
            pvals = 1.- sp.stats.gamma.cdf(stat, a, scale=b)
        else:
            residTup = utils.levene_pval(residual, n_ex_valid, num_tasks)
            pvals = sp.stats.levene(*residTup)[1]
        
        all_sets.append(s)
        all_pvals.append(pvals)
                                                                      
        if (pvals > alpha[0]):
            mse_current = np.mean((pred - valid_y) ** 2)
            if mse_current < best_mse: 
                best_mse = mse_current
                best_subset = s
                current_inter = np.intersect1d(current_inter, s)
                accepted_sets.append(s)
                accepted_mse.append(mse_current)

if len(accepted_sets) == 0:
    all_pvals = np.array(all_pvals).flatten()
    sort_pvals = np.argsort(all_pvals)
    idx_max = sort_pvals[-1]
    best_subset = all_sets[idx_max]
    accepted_sets.append(best_subset)

np.array(best_subset)               
print(time.time() - start)
X_train = X_train_data[best_subset]
X_test = X_test_data[best_subset]

##############

#subset = subsetArg(y=2,s=5)
#print(subset)


#n_estimators = numBootstraps(1000)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
model.fit(X_train,y_train_data)
y_pred = model.predict(X_test)

print(sum((y_pred-y_test_data)**2))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_pred,y_test_data))
