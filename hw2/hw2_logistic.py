# Machine Learning 2017
# Hw2 : Classification
# Logistic model (deterministic)

import numpy as np
#import random as rnd
import sys
import csv

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def myfloat(a):
    if isfloat(a):
        if ( float(a) > 0):
            return float(a)
        else:
            return 0.0
    else:
        return 0.0
i = 0
with open(sys.argv[1],'r',encoding='big5') as csvFile:
    for row in csv.reader(csvFile):
        if i == 0:
            print(list(map(myfloat,row)))
            i = i + 1
