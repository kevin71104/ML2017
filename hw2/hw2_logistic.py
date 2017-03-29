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

####################### Read X_train ########################
train = []
with open(sys.argv[1],'r') as csvFile:
    for row in csv.reader(csvFile):
        train.append( list( map(myint,row) ) )
# retrieve first row
train  = np.array(train[1:])
person = train.shape[0]  # number of person
num_w  = train.shape[1]  # number of weight

####################### Read Y_train ########################
label = []
with open(sys.argv[2],'r') as csvFile:
    for row in csv.reader(csvFile):
        #print(row)
        label.extend( list( map(myint,row) ) )
#print(label)
label = np.array(label)

####################### Normalization #######################
mean  = train.mean(0)
std   = train.std(0)
train = (train - mean) / std

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

# Regularization

# Record Parameters
his = [[b] + w.tolist()]

####################### Start Training ######################
for traintime in range(1000):
    loss = 0.0
    wgrad.fill(0.0)
    bgrad = 0.0
    z = np.sum(w * train, axis = 1) + b
    f = 1 / (1 + np.exp(-z))
    f[f == 0] = 1e-8
    f[f == 1] = 1-1e-8
    loss = -(label*np.log(f) + (1-label)*np.log((1-f)) )
    temp = f - label
    bgrad = np.sum(temp)
    wgrad = np.dot(temp,train)  # temp:1*m  train: m*n

    wsum = wsum + wgrad**2
    bsum = bsum + bgrad**2

    w = w - (wlr/np.sqrt(wsum) )* wgrad
    b = b - (blr/np.sqrt(bsum) )* bgrad

    his.append([b]+w.tolist())

###################### Write Parameters #####################
with open(sys.argv[5],'w') as csvFile:
    for row in range(len(his[-1])):
        csvFile.write(str(row)+',' + str(his[-1][row])+'\n')

####################### Read X_test #########################
test = []
with open(sys.argv[3],'r') as csvFile:
    for row in csv.reader(csvFile):
        test.append( list( map(myint,row) ) )
# retrieve first row
test = np.array(test[1:])
#for i in range(10):
#    print(test[i])
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
