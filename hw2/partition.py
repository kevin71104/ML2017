#############################################################
#                   Machine Learning 2017                   #
#             Partition data into train & validation        #
#############################################################

import numpy as np
import sys
import csv

def myint(a):
    if a.isdigit() :
        return int(a)
    else:
        return 0

####################### Read data ########################
train = []
temp = 0
with open(sys.argv[1],'r') as csvFile:
    for row in csv.reader(csvFile):
        if temp == 0:
            title = row
            temp = 1
        train.append( list( map(myint,row) ) )
# retrieve first row
train  = np.array(train[1:])
person = train.shape[0]  # number of person
num_w  = train.shape[1]  # number of weight

####################### write data ########################
div = int(person/3)
part = train[:(div+1)]
with open(sys.argv[2],'w') as csvFile:
    for i in range(len(title)):
        if i == 0:
            csvFile.write(title[i])
        else:
            csvFile.write(',' + title[i])
    for row in part:
        csvFile.write('\n')
        for i in range(len(row)):
            if i == 0:
                csvFile.write(str(row[i]))
            else:
                csvFile.write(',' + str(row[i]) )

####################### Read Y_train ########################
label = []
with open(sys.argv[2],'r') as csvFile:
    for row in csv.reader(csvFile):
        #print(row)
        label.extend( list( map(myint,row) ) )
#print(label)
label = np.array(label)
