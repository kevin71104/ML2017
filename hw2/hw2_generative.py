#############################################################
#                   Machine Learning 2017                   #
#                   Hw2 : Classification                    #
#             Generative model (deterministic)              #
#############################################################

import numpy as np
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
train  = train[1:]
person = len(train)  # number of person
num_w  = len(train[0])  # number of weight

####################### Read Y_train ########################
label = []
with open(sys.argv[2],'r') as csvFile:
    for row in csv.reader(csvFile):
        label.extend( list( map(myint,row) ) )

#################### Classify Train_data ####################
highpay = []
lesspay = []
for i in range(person):
    if label[i]:
        highpay.append(train[i])
    else:
        lesspay.append(train[i])
#for i in range(10):
#    print(highpay[i])
