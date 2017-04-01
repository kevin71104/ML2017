#############################################################
#                   Machine Learning 2017                   #
#                   Hw2 : Classification                    #
#                       validization                        #
#############################################################

import numpy as np
import sys
import csv

def myint(a):
    if a.isdigit() :
        return int(a)
    else:
        return 0

myans = []
with open(sys.argv[1],'r') as csvFile:
    for row in csv.reader(csvFile):
        #print(row)
        myans.extend( list( map(myint,row) ) )

label = []
with open(sys.argv[2],'r') as csvFile:
    for row in csv.reader(csvFile):
        #print(row)
        label.extend( list( map(myint,row) ) )

count = 0
for i in range(len(myans)):
    if myans[i] == label[i]:
        count = count+1
print(float(count)/len(myans))
