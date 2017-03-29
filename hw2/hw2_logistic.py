############################################################
#                  Machine Learning 2017                   #
#                  Hw2 : Classification                    #
#             Logistic model (deterministic)               #
############################################################

import numpy as np
#import random as rnd
import sys
import csv

def myint(a):
    if a.isdigit() :
        return int(a)
    else:
        return 0

#################### Read X_train ####################
train = []
with open(sys.argv[1],'r') as csvFile:
    for row in csv.reader(csvFile):
        train.append( list( map(myint,row) ) )
# retrieve first row
train = np.array(train[1:])

#################### Read Y_train ####################
label = []
with open(sys.argv[2],'r') as csvFile:
    for row in csv.reader(csvFile):
        #print(row)
        label.extend( list( map(myint,row) ) )
#print(label)
label = np.array(label)

#################### Normalization ####################
mean  = train.mean(0)
std   = train.std(0)
train = (train - mean) / std

################ Initialize Parameters ################

#################### Start Training ###################

##################### Read X_test #####################
test = []
with open(sys.argv[3],'r') as csvFile:
    for row in csv.reader(csvFile):
        test.append( list( map(myint,row) ) )
# retrieve first row
test = np.array(test[1:])
for i in range(10):
    print(test[i])
test = (test - mean) / std

##################### Write Y_test ####################
