import numpy as np
import sys
import csv

answer = []
op = []
with open(sys.argv[1],'r') as csvFile:
    for row in csvFile:
        if(row[1] != 'value'):
            op.extend(row[1])
with open(sys.argv[2],'r') as csvFile:
    for row in csvFile:
        if(row[1] != 'value'):
            answer.extend(row[1])
diff = np.array(op)-np.array(answer)
print(np.inner(diff,diff))
