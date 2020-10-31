# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:56:59 2020

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
n_tasks = 18
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
    p = t + np.random.normal(0, sx2, (n_examples_task, 1))
    
    x_task = np.concatenate([c1,c2,x,y,p,c1,c2,x,y,p,c1,c2,x,y,p,c1,c2,x],axis = 1)
    train_x = np.append(train_x, x_task, axis = 0)
    train_y = np.append(train_y, t)
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
valid_split = 0.1
n_ex_cum = np.append(0, np.cumsum(n_ex))
x, y = train_x, train_y 
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

import time
start = time.time()
num_s = np.sum(n_ex)

num_predictors = train_x.shape[1]
best_subset = np.array([])
best_subset_acc = np.array([])
best_mse_overall = 1e10

already_acc = False

selected = np.zeros(num_predictors)
accepted_subset = None

all_sets, all_pvals = [], []

n_iters = 10*num_predictors
stay = 1

pow_2 = np.array([2**i for i in np.arange(num_predictors)])

ind = 0
prev_stat = 0

bins = []
   
#Get numbers for the mean

pred = np.mean(train_y)
mse_current = np.mean((pred - valid_y) ** 2)
residual = valid_y - pred

residTup = utils.levene_pval(residual, n_ex_valid, n_ex_valid.size)
#residTup = np.array(residTup).flatten()
residTup = [np.array(sublist).ravel() for sublist in residTup]
levene = sp.stats.levene(*residTup)

all_sets.append(np.array([]))
all_pvals.append(levene[1])
#if all_pvals[-1]>alpha:
#  accepted_subset = np.array([])

while (stay==1):
    
    pvals_a = np.zeros(num_predictors)
    statistic_a = 1e10 * np.ones(num_predictors)
    mse_a = np.zeros(num_predictors)

    for p in range(num_predictors):
        current_subset = np.sort(np.where(selected == 1)[0])
        regr = linear_model.LinearRegression()
        
        if selected[p]==0:
            subset_add = np.append(current_subset, p).astype(int)
            regr.fit(train_x[:,subset_add], train_y.flatten())
            
            pred = regr.predict(valid_x[:,subset_add])[:,np.newaxis]
            mse_current = np.mean((pred - valid_y)**2)
            residual = valid_y - pred

            residTup = utils.levene_pval(residual,n_ex_valid,
                                                 n_ex_valid.size)
            
            residTup = [np.array(sublist).ravel() for sublist in residTup]
            levene = sp.stats.levene(*residTup)

            pvals_a[p] = levene[1]
            statistic_a[p] = levene[0]
            mse_a[p] = mse_current

            all_sets.append(subset_add)
            all_pvals.append(levene[1])
            
        if selected[p] == 1:
            acc_rem = np.copy(selected)
            acc_rem[p] = 0

            subset_rem = np.sort(np.where(acc_rem == 1)[0])

            if subset_rem.size ==0: continue
            
            regr = linear_model.LinearRegression()
            regr.fit(train_x[:,subset_rem], train_y.flatten())

            pred = regr.predict(valid_x[:,subset_rem])[:,np.newaxis]
            mse_current = np.mean((pred - valid_y)**2)
            residual = valid_y - pred
            
            residTup = utils.levene_pval(residual,n_ex_valid, 
                                                 n_ex_valid.size)
            
            residTup = [np.array(sublist).ravel() for sublist in residTup]
            levene = sp.stats.levene(*residTup)
            
            pvals_a[p] = levene[1]
            statistic_a[p] = levene[0]
            mse_a[p] = mse_current

            all_sets.append(subset_rem)
            all_pvals.append(levene[1])

    accepted = np.where(pvals_a > alpha)

    if accepted[0].size>0:
        best_mse = np.amin(mse_a[np.where(pvals_a > alpha)])
        already_acc = True

        selected[np.where(mse_a == best_mse)] = \
          (selected[np.where(mse_a == best_mse)] + 1) % 2

        accepted_subset = np.sort(np.where(selected == 1)[0])
        binary = np.sum(pow_2 * selected)
   
        if (bins==binary).any():
            stay = 0
        bins.append(binary)
    else:
        best_pval_arg = np.argmin(statistic_a)

        selected[best_pval_arg] = (selected[best_pval_arg] + 1) % 2
        binary = np.sum(pow_2 * selected)

        if (bins==binary).any():
            stay = 0
        bins.append(binary)

    if ind>n_iters:
        stay = 0
    ind += 1

if accepted_subset is None:
  all_pvals = np.array(all_pvals).flatten()

  max_pvals = np.argsort(all_pvals)[-1]
  accepted_subset = np.sort(all_sets[max_pvals])

np.array(accepted_subset)
print(time.time() -start)

