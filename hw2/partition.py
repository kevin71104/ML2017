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
        train.append(row)
title = train[0]
train = train[1:]

####################### write data ########################
div = int (len(train)/3)
#print(div)
part1 = train[:div]
part2 = train[div:div*2]
part3 = train[div*2:]
with open('x1.csv','w') as csvFile:
    for i in range(len(title)):
        if i == 0:
            csvFile.write(title[i])
        else:
            csvFile.write(',' + title[i])
    for row in part1:
        csvFile.write('\n')
        for i in range(len(row)):
            if i == 0:
                csvFile.write(row[i])
            else:
                csvFile.write(',' + row[i])
with open('x2.csv','w') as csvFile:
    for i in range(len(title)):
        if i == 0:
            csvFile.write(title[i])
        else:
            csvFile.write(',' + title[i])
    for row in part2:
        csvFile.write('\n')
        for i in range(len(row)):
            if i == 0:
                csvFile.write(row[i])
            else:
                csvFile.write(',' + row[i])
with open('x3.csv','w') as csvFile:
    for i in range(len(title)):
        if i == 0:
            csvFile.write(title[i])
        else:
            csvFile.write(',' + title[i])
    for row in part3:
        csvFile.write('\n')
        for i in range(len(row)):
            if i == 0:
                csvFile.write(row[i])
            else:
                csvFile.write(',' + row[i])
temp = part1 + part2
with open('x12.csv','w') as csvFile:
    for i in range(len(title)):
        if i == 0:
            csvFile.write(title[i])
        else:
            csvFile.write(',' + title[i])
    for row in temp:
        csvFile.write('\n')
        for i in range(len(row)):
            if i == 0:
                csvFile.write(row[i])
            else:
                csvFile.write(',' + row[i])
temp = part1 + part3
with open('x13.csv','w') as csvFile:
    for i in range(len(title)):
        if i == 0:
            csvFile.write(title[i])
        else:
            csvFile.write(',' + title[i])
    for row in temp:
        csvFile.write('\n')
        for i in range(len(row)):
            if i == 0:
                csvFile.write(row[i])
            else:
                csvFile.write(',' + row[i])
temp = part2 + part3
with open('x23.csv','w') as csvFile:
    for i in range(len(title)):
        if i == 0:
            csvFile.write(title[i])
        else:
            csvFile.write(',' + title[i])
    for row in temp:
        csvFile.write('\n')
        for i in range(len(row)):
            if i == 0:
                csvFile.write(row[i])
            else:
                csvFile.write(',' + row[i])

####################### Read Y_train ########################
label = []
with open(sys.argv[2],'r') as csvFile:
    for row in csv.reader(csvFile):
        label.extend(row)
####################### write Y_train #######################
part1 = label[:div]
part2 = label[div:div*2]
part3 = label[div*2:]

x = 1
with open('y1.csv','w') as csvFile:
    csvFile.write('id,label')
    for row in part1:
        csvFile.write('\n' + str(x) + ',' + row)
        x = x+1

x = 1
with open('y2.csv','w') as csvFile:
    csvFile.write('id,label')
    for row in part2:
        csvFile.write('\n' + str(x) + ',' + row)
        x = x+1

x = 1
with open('y3.csv','w') as csvFile:
    csvFile.write('id,label')
    for row in part3:
        csvFile.write('\n' + str(x) + ',' + row)
        x = x+1

temp = part1 + part2
with open('./partition/y12.csv','w') as csvFile:
    for i in range(len(temp)):
        if i == 0:
            csvFile.write(temp[i])
        else:
            csvFile.write('\n' + temp[i])

temp = part1 + part3
with open('./partition/y13.csv','w') as csvFile:
    for i in range(len(temp)):
        if i == 0:
            csvFile.write(temp[i])
        else:
            csvFile.write('\n' + temp[i])

temp = part2 + part3
with open('./partition/y23.csv','w') as csvFile:
    for i in range(len(temp)):
        if i == 0:
            csvFile.write(temp[i])
        else:
            csvFile.write('\n' + temp[i])
