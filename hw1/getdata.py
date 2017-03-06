import numpy as np
import sys
import csv

# get train data
train_data = []
with open(sys.argv[1],'r',encoding='big5') as csvFile:
    for row in csv.reader(csvFile):
        if(row[2] == 'PM2.5'):
             train_data.extend( list( map(int,row[3:27]) ) )

# store last 240 data
test = []
answer = []
for i in range(240):
    test.append(train_data[(3360+10*i):(3360+10*i+9)])
    answer.append(train_data[3360+10*i+9])
ID = 0
with open('test.csv','w') as csvFile:
    for row in test:
        if(ID != 0):
            csvFile.write('\n')
        csvFile.write('id_'+str(ID)+',PM2.5')
        for i in row:
            csvFile.write(','+str(i))
        ID = ID +1
ID = 0
with open('ans.csv','w') as csvFile:
    csvFile.write('id,value')
    for row in answer:
        csvFile.write('\nid_' + str(ID) + ',' + str(row))
        ID = ID+1
