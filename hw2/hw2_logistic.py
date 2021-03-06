#############################################################
#                   Machine Learning 2017                   #
#                   Hw2 : Classification                    #
#              Logistic model (deterministic)               #
#############################################################

import numpy as np
#import random as rnd
import sys
import csv

def myint(a):
    if a.isdigit() :
        return int(a)
    else:
        return 0

def mytwo(a):
    return a**2

def mythree(a):
    return a**3

def myfour(a):
    return (a**4/10000)

######################## Read train #########################
if len(sys.argv) > 6 :
    extra = []
    with open(sys.argv[6],'r') as csvFile:
        for row in csv.reader(csvFile):
            extra.append(int(row[4]))
#print (extra)

####################### Read X_train ########################
train = []
with open(sys.argv[1],'r') as csvFile:
    for row in csv.reader(csvFile):
        train.append( list( map(myint,row) ) )
# retrieve first row
train = train[1:]
for i in range(len(train)):
    temp = list(map(mytwo,(train[i][0:2]+train[i][3:6])))   + \
           list(map(mythree,(train[i][0:2]+train[i][3:6])))
    """temp = list(map(mytwo,train[i][0:6]))   + \
           list(map(mythree,train[i][0:6]))
           #list(map(myfour,train[i][0:6]))
           """
    train[i].extend(temp)
    # add education_num
    #train[i].extend([extra[i], extra[i]**2, extra[i]**3, extra[i]**4])

train  = np.array(train)
person = train.shape[0]
num_w  = train.shape[1]

###################### Read Y_train #########################
label = []
with open(sys.argv[2],'r') as csvFile:
    for row in csv.reader(csvFile):
        label.extend( list( map(myint,row) ) )
label = np.array(label)

####################### Normalization #######################
mean  = train.mean(0)
std   = train.std(0)
std[std == 0.0] = 1 # std = 0
#print(std)
train = (train - mean) / std
#print (train[0])

################### Initialize Parameters ###################
# Weight and its learning rate
w        = np.zeros_like(train[0,:], dtype = float)
wlr      = np.zeros_like(train[0,:], dtype = float)
wsum     = np.zeros_like(train[0,:], dtype = float)
wgrad    = np.zeros_like(train[0,:], dtype = float)
wlr.fill(0.5)
wsum.fill(1e-8)

# Bias and its learning rate
b     = 0.0
blr   = 0.1
bsum  = 1e-8
bgrad = 0.0
"""
# Get previous model
model = []
with open('bestmodel.csv','r') as csvFile:
    for row in csv.reader(csvFile):
        model.append(float(row[1]))
"""

# Regularization
lamb = 0.1

# Record Parameters
his = [[b] + w.tolist()]

####################### Start Training ######################
for traintime in range(10000):
    loss = 0.0
    wgrad.fill(0.0)
    bgrad = 0.0
    z = np.sum(w * train, axis = 1) + b
    f = 1 / (1 + np.exp(-z))
    f[f == 0] = 1e-10
    f[f == 1] = 1-1e-10
    loss = -(label*np.log(f) + (1-label)*np.log((1-f)) ) + np.sum(w)*lamb
    print( str(traintime) + ' ' + str(np.sum(loss)/person))
    temp = f - label
    bgrad = np.sum(temp)
    wgrad = np.dot(temp,train)  # temp:1*m  train: m*n

    wsum = wsum + wgrad**2
    bsum = bsum + bgrad**2

    w = w - (wlr/np.sqrt(wsum) )* wgrad
    b = b - (blr/np.sqrt(bsum) )* bgrad

    his.append([b]+w.tolist())

###################### Write Parameters #####################
if len(sys.argv) > 5 :
    storemodel = his[-1]+[bsum]+wsum.tolist()
    #storemodel = his[-1]
    with open(sys.argv[5],'w') as csvFile:
        for row in range(len(storemodel)):
            csvFile.write(str(row)+',' + str(storemodel[row])+'\n')

######################## Read test ##########################
if len(sys.argv) > 7 :
    extra = []
    with open(sys.argv[7],'r') as csvFile:
        for row in csv.reader(csvFile):
            extra.append(myint(row[4]))
    extra = extra[1:]
#print (extra)

####################### Read X_test #########################
test = []
with open(sys.argv[3],'r') as csvFile:
    for row in csv.reader(csvFile):
        test.append( list( map(myint,row) ) )
# retrieve first row
test = test[1:]
for i in range(len(test)):
    temp = list(map(mytwo,(test[i][0:2]+test[i][3:6])))   + \
           list(map(mythree,(test[i][0:2]+test[i][3:6])))
    """temp = list(map(mytwo,test[i][0:6]))   + \
           list(map(mythree,test[i][0:6]))
           #list(map(myfour,test[i][0:6]))"""
    test[i].extend(temp)
    # add education_num
    #test[i].extend([extra[i],extra[i]**2,extra[i]**3,extra[i]**4])
test  = np.array(test)
test = (test - mean) / std

######################## Write Y_test #######################
output = []
for i in range(test.shape[0]):
    if (np.inner(w,test[i])+b) > 0:
        output.append(1)
    else:
        output.append(0)

x = 1
with open(sys.argv[4],'w') as csvFile:
    csvFile.write('id,label')
    for row in output:
        csvFile.write('\n' + str(x) + ',' + str(row))
        x = x+1
