# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:08:30 2020

@author: Om Pandey
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

def readData(y,i):
    input_file = r"C:\Users\KIIT\Desktop\Causal\data\output1.csv"
    Y = y
    I = i
    #S = int(sys.argv[4])
    currentDir = r"C:\Users\KIIT\Desktop\Causal\data"

    input_data = pd.read_csv(input_file)

    # Read metadata file (for numbers of variables and blinding)
    is_split = False
    li = input_file.rsplit('real-', 1)
    if len(li) > 1:
        is_split = True
    meta_file = 'meta-'.join(li)
    li = meta_file.rsplit('nan-', 1)
    if len(li) > 1:
        is_split = True
    meta_file = 'meta-'.join(li)
    if is_split and os.path.isfile(meta_file):
        metadata_present = True
        metadata = pd.read_csv(meta_file)
        num_sys_vars = metadata.loc[0,'num_sys_vars']
        num_ints_vars = metadata.loc[0,'num_ints_vars']
        IBlind = metadata.loc[0,'IBlind']
        IBlind_visible_value = metadata.loc[0,'IBlind_visible_value']
    else:
        metadata_present = False

    if not metadata_present or input_data.columns.size == num_sys_vars + 1:
        # Context variables not provided in data files: construct a diagonal design
        regime_column = input_data.columns[input_data.columns.size-1]

        # Construct train and test data
        unknown_rows = input_data[regime_column].isin([I+1])

        # Create the experimental design with diagonal matrix and then ignore the regime:
       # dummy_data = pd.get_dummies(input_data[regime_column],drop_first=True) # drop_first not on cluster
        dummy_data_withfirst = pd.get_dummies(input_data[regime_column])
        data = pd.concat([input_data, dummy_data_withfirst.iloc[:,1:]], axis=1)
    elif input_data.columns.size == num_sys_vars + 1 + num_ints_vars:
        # Context variables are provided in data files

        # The test set consists of those rows where context variable IBlind's value is *not* equal to
        # IBlind_visible_value
        IBlind_column = input_data.columns[num_sys_vars+IBlind] # not +1 because Python uses 0-based indexing
        unknown_rows = ~(input_data[IBlind_column].isin([IBlind_visible_value]))

        # No need to add any columns
        data = input_data
    else:
        print ("Error: expected either "+str(num_sys_vars + 1)+" (sys+R) or "+str(num_sys_vars + 1 + num_ints_vars)+
               "(sys+R+context) columns; got "+str(input_data.columns.size))
        sys.exit()

    #print "training points: ", y_train_data.shape[0]
    #print "test points: ", y_test_data.shape[0]

    data_unknown_rows = data.loc[unknown_rows]
    data_known_rows = data.loc[~unknown_rows]

    X_test_data = data_unknown_rows.copy()
    y_test_data = data_unknown_rows[data_known_rows.columns[Y-1]].copy()
    X_train_data = data_known_rows.copy()
    
    y_train_data = data_known_rows[data_known_rows.columns[Y-1]].copy()

    return X_test_data, y_test_data, X_train_data, y_train_data

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


def outputFilename(y,i,s,tag):
    input_file = r"C:\Users\KIIT\Desktop\Causal\data\output1.csv"
    Y = y
    I = i
    S = s
    currentDir = r"C:\Users\KIIT\Desktop\Causal\data"
    
    i = input_file.split("-")[1].split(".")[0]
    return os.path.join(currentDir, tag + "-" + i + "-" + str(Y) + "-" + str(I) + "-" + str(S) + ".csv")

def readData1(y):
    le = LabelEncoder()
    input_file = r"C:\Users\KIIT\Desktop\Causal\data\output_dis_2.csv"
    input_data = pd.read_csv(input_file)
    input_data = input_data.iloc[:,1:]
    input_data = input_data.apply(le.fit_transform)
    test_data = input_data[input_data.C1 == 1]  
    train_data = input_data[~(input_data.C1 == 1)]
    y_train = list(train_data["T"])
    y_test = list(test_data["T"])
    train_data = train_data.drop(["T"], axis =1)
    test_data = test_data.drop(["T"], axis =1)
    return test_data, y_test, train_data, y_train

X_test_data, y_test_data, X_train_data, y_train_data = readData1(y=16)
################
ls = ["Y"] #ls = list(X_train_data.columns) 
"""This(ls) is best subset selected based on performance of test, we have hardcoded but you can import subset
by saving as csv from R for large dataset"""
X_train = X_train_data[ls]
X_test = X_test_data[ls]

##############

#subset = subsetArg(y=2,s=5)
#print(subset)

import time
start = time.time()
n_estimators = numBootstraps(1000)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
model.fit(X_train,y_train_data)
y_pred = model.predict(X_test)

print(sum((y_pred-y_test_data)**2))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_pred,y_test_data))
print(start - time.time())

baseline=[ 2.3363996529901856,66.65546324274047,1.8000000000129262e-07,0.004764469999998574]

meth=[3.6947618320029987, 77.71301923465275, 1.8000000000129262e-07,0.004764469999998574]

import seaborn as sns
sns.set(style="whitegrid")
tot=[baseline,meth]
columns = ["base","meth"]
ax = sns.boxplot(x = tot, y= columns)


#Here y_train column should not be in subset
subset = [0,13,17,18]
n_estimators = numBootstraps(1000)
import time
start_time = time.time()
D = RegressionAllSubsets(X_train_data,y_train_data,subset,n_estimators)
print("--- %s seconds ---" % (time.time() - start_time))


time = {"nbr" : 12.041055679321289, "mb": 29.89686393737793}
set_list = D["Set"].tolist()

int_list = map(SubsetToInt, set_list)
    
print(list(int_list))

output_file =  outputFilename(2,2,5,"fs")

with open(output_file, 'w') as output:
    wr = csv.writer(output)
    wr.writerow(set_list)


input_file = r"C:\Users\KIIT\Desktop\Causal\data\output_dis_2.csv"
input_data = pd.read_csv(input_file)
input_data = input_data.iloc[:,1:]
test_data = input_data[input_data.C1 == 'a']  
train_data = input_data[~(input_data.C1 == 'a')]